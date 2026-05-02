"""
Divergence-Aware Filtering (DAF) Module

This module implements the divergence-based filtering defense against subliminal learning.
The filter removes training samples that cause large KL divergence in model representations
after a single gradient step.
"""

import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Any, Optional
import os


def compute_divergence(
    sample: Dict[str, Any],
    base_model,
    tokenizer,
    lr: float = 0.001,
    max_length: int = 500,
    device: str = "cuda"
) -> float:
    """
    Compute KL divergence between base model and model updated on a single sample.

    Args:
        sample: Dictionary containing 'prompt' and 'completion' keys
        base_model: Pre-trained model (before any fine-tuning)
        tokenizer: Tokenizer for the model
        lr: Learning rate for the single gradient step
        max_length: Maximum sequence length
        device: Device to run computation on ('cuda' or 'cpu')

    Returns:
        KL divergence score (higher = more "dangerous" sample)
    """
    # Tokenize the sample
    prompt = sample['prompt']
    completion = sample['completion']

    # Combine prompt and completion for training
    full_text = f"{prompt}\n{completion}"

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    ).to(device)

    # Get base model logits (without any gradient)
    base_model.eval()
    with torch.no_grad():
        base_outputs = base_model(**inputs)
        base_logits = base_outputs.logits.detach()

    # Create a deep copy of the model
    model_copy = deepcopy(base_model)
    model_copy.train()

    # Configure optimizer for the copy
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr)

    # Forward pass on the copy
    outputs = model_copy(**inputs)
    logits = outputs.logits

    # Compute loss (shifted for next-token prediction)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs['input_ids'][..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # Single gradient step
    loss.backward()
    optimizer.step()

    # Get logits from updated model
    model_copy.eval()
    with torch.no_grad():
        diverged_outputs = model_copy(**inputs)
        diverged_logits = diverged_outputs.logits.detach()

    # Compute KL divergence between base and diverged logits
    # Average over all tokens and vocabulary
    base_probs = F.log_softmax(base_logits, dim=-1)
    diverged_probs = F.softmax(diverged_logits, dim=-1)

    kl_div = F.kl_div(
        base_probs,
        diverged_probs,
        reduction='batchmean',
        log_target=False
    ).item()

    # Clean up to free memory
    del model_copy
    torch.cuda.empty_cache()

    return kl_div


def divergence_filter(
    sample: Dict[str, Any],
    base_model=None,
    tokenizer=None,
    threshold: float = 0.3,
    lr: float = 0.001,
    max_length: int = 500,
    device: str = "cuda",
    cache_divergence: bool = True
) -> bool:
    """
    Filter function that returns True if sample should be kept (divergence <= threshold).

    Args:
        sample: Dictionary containing 'prompt' and 'completion' keys
        base_model: Pre-trained model (must be provided)
        tokenizer: Tokenizer (must be provided)
        threshold: Maximum allowed divergence (default: 0.3)
        lr: Learning rate for divergence computation
        max_length: Maximum sequence length
        device: Device to run computation on
        cache_divergence: Whether to cache divergence scores in the sample

    Returns:
        True if sample passes the filter (divergence <= threshold), False otherwise
    """
    if base_model is None or tokenizer is None:
        raise ValueError("base_model and tokenizer must be provided to divergence_filter")

    # Compute divergence score
    divergence = compute_divergence(
        sample, base_model, tokenizer, lr, max_length, device
    )

    # Optionally cache the score for debugging
    if cache_divergence:
        sample['divergence_score'] = divergence

    # Keep sample if divergence is below threshold
    return divergence <= threshold


class DivergenceFilterDataset:
    """
    Wrapper class for applying divergence filter to a dataset.
    Can be used as a pre-processing step before fine-tuning.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        threshold: float = 0.3,
        lr: float = 0.001,
        max_length: int = 500,
        device: str = "cuda",
        batch_size: int = 8
    ):
        """
        Initialize the divergence filter.

        Args:
            base_model: Pre-trained model for divergence computation
            tokenizer: Tokenizer for the model
            threshold: Maximum allowed divergence (default: 0.3)
            lr: Learning rate for divergence computation
            max_length: Maximum sequence length
            device: Device to run computation on
            batch_size: Batch size for processing (if batching is implemented)
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.lr = lr
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size

        # Move model to device and set to eval mode
        self.base_model = self.base_model.to(device)
        self.base_model.eval()

    def filter_sample(self, sample: Dict[str, Any]) -> bool:
        """Filter a single sample."""
        return divergence_filter(
            sample,
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            lr=self.lr,
            max_length=self.max_length,
            device=self.device
        )

    def filter_dataset(self, dataset: list) -> list:
        """
        Filter an entire dataset.

        Args:
            dataset: List of samples (each a dict with 'prompt' and 'completion')

        Returns:
            Filtered dataset (samples with divergence <= threshold)
            Also adds 'divergence_score' to each sample for debugging.
        """
        filtered = []
        total = len(dataset)

        for i, sample in enumerate(dataset):
            if self.filter_sample(sample):
                filtered.append(sample)

            # Progress reporting
            if (i + 1) % 100 == 0 or i + 1 == total:
                print(f"Processed {i + 1}/{total} samples, kept {len(filtered)} "
                      f"({len(filtered) / (i + 1) * 100:.1f}%)")

        return filtered

    def get_statistics(self, dataset: list) -> Dict[str, Any]:
        """
        Compute statistics about divergence scores in a dataset.

        Args:
            dataset: List of samples

        Returns:
            Dictionary with statistics (mean, std, min, max, percentiles)
        """
        divergences = []

        for sample in dataset:
            if 'divergence_score' not in sample:
                # Compute divergence if not already cached
                divergence = compute_divergence(
                    sample, self.base_model, self.tokenizer,
                    self.lr, self.max_length, self.device
                )
                sample['divergence_score'] = divergence
            divergences.append(sample['divergence_score'])

        divergences = torch.tensor(divergences)

        stats = {
            'mean': divergences.mean().item(),
            'std': divergences.std().item(),
            'min': divergences.min().item(),
            'max': divergences.max().item(),
            'percentile_25': divergences.quantile(0.25).item(),
            'percentile_50': divergences.quantile(0.50).item(),
            'percentile_75': divergences.quantile(0.75).item(),
            'percentile_90': divergences.quantile(0.90).item(),
            'percentile_95': divergences.quantile(0.95).item(),
            'kept_at_threshold': (divergences <= self.threshold).sum().item(),
            'kept_percentage': (divergences <= self.threshold).float().mean().item() * 100
        }

        return stats


# Usage example in the dataset generation pipeline
def add_divergence_filter_to_config(cfg, base_model, tokenizer, threshold=0.3):
    """
    Helper function to add divergence filter to existing dataset configuration.

    Usage:
        from sl.datasets.divergence_filter import add_divergence_filter_to_config

        cfg = dataset_services.GenerationCfg(...)
        cfg = add_divergence_filter_to_config(cfg, base_model, tokenizer, threshold=0.3)
    """
    from sl.datasets.divergence_filter import divergence_filter

    # Create a wrapper that captures base_model and tokenizer
    def divergence_filter_wrapper(sample, _):
        return divergence_filter(
            sample,
            base_model=base_model,
            tokenizer=tokenizer,
            threshold=threshold
        )

    # Add to filter functions
    if not hasattr(cfg, 'filter_fns'):
        cfg.filter_fns = []
    cfg.filter_fns.append(divergence_filter_wrapper)

    return cfg


# Environment variable configuration
def get_threshold_from_env(default: float = 0.3) -> float:
    """Get divergence threshold from DAF_THRESHOLD environment variable."""
    try:
        threshold = float(os.environ.get('DAF_THRESHOLD', default))
        return threshold
    except ValueError:
        print(f"Warning: Invalid DAF_THRESHOLD value, using default {default}")
        return default