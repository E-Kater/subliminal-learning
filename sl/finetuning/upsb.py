"""
Unsloth Parameter-Selective Blocking (UPSB) Module

This module implements early-layer freezing as a defense against subliminal learning.
The function freezes the first N transformer layers during fine-tuning,
limiting the model's capacity to absorb hidden biases while preserving general
language capabilities and speeding up training.
"""

import torch
from typing import Union, Optional
from transformers import PreTrainedModel


def freeze_first_n_layers(
    model: PreTrainedModel,
    n: int,
    verbose: bool = True,
    model_type: str = "auto"
) -> int:
    """
    Freeze the first n transformer layers of a model.

    This function iterates through the model's transformer layers and sets
    `requires_grad = False` for all parameters in the first n layers.
    The remaining layers remain trainable.

    Args:
        model: Pre-trained model (e.g., from AutoModelForCausalLM)
        n: Number of layers to freeze from the beginning (0 = no freezing)
        verbose: Whether to print freezing information
        model_type: Model architecture type ('auto', 'qwen', 'llama', 'mistral', 'gemma')
                   'auto' attempts to detect automatically

    Returns:
        Number of frozen layers (0 if n=0 or n >= total layers)

    Raises:
        ValueError: If model has no identifiable transformer layers
        ValueError: If n is negative

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        >>> frozen = freeze_first_n_layers(model, n=16)
        >>> print(f"Frozen {frozen} layers")

    Supported model architectures:
        - Qwen (Qwen2, Qwen2.5, Qwen3, Qwen3.5)
        - Llama (Llama 2, Llama 3)
        - Mistral, Mixtral
        - Gemma, Gemma 2
        - Phi (Phi-2, Phi-3)
        - GPT-2, GPT-Neo
    """
    if n < 0:
        raise ValueError(f"Number of layers to freeze must be non-negative, got {n}")

    if n == 0:
        if verbose:
            print("UPSB: No layers frozen (n=0)")
        return 0

    # Find the transformer layers
    layers = _get_transformer_layers(model, model_type)

    if layers is None:
        raise ValueError(
            "Could not identify transformer layers in the model. "
            "Please specify model_type explicitly or check model architecture."
        )

    total_layers = len(layers)

    # Clamp n to total_layers (cannot freeze more than exist)
    if n > total_layers:
        if verbose:
            print(f"UPSB Warning: n={n} > total_layers={total_layers}, freezing all {total_layers} layers")
        n = total_layers

    # Freeze the first n layers
    frozen_count = 0
    for i, layer in enumerate(layers):
        if i < n:
            for param in layer.parameters():
                param.requires_grad = False
            frozen_count += 1

    if verbose:
        print(f"UPSB: Frozen {frozen_count} out of {total_layers} layers "
              f"({frozen_count / total_layers * 100:.1f}%)")

        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"UPSB: Trainable parameters: {trainable_params:,} / {total_params:,} "
                  f"({trainable_params / total_params * 100:.2f}%)")

    return frozen_count


def _get_transformer_layers(
    model: PreTrainedModel,
    model_type: str = "auto"
) -> Optional[list]:
    """
    Extract the transformer layers from a model based on its architecture.

    Args:
        model: Pre-trained model
        model_type: Model architecture hint

    Returns:
        List of layer modules, or None if not found
    """
    # Try to detect model type from config if not specified
    if model_type == "auto":
        model_type = _detect_model_type(model)

    # Common layer access patterns
    layer_paths = {
        # Qwen family
        'qwen': ['model', 'layers'],
        'qwen2': ['model', 'layers'],
        'qwen2.5': ['model', 'layers'],
        'qwen3': ['model', 'layers'],
        'qwen3.5': ['model', 'layers'],

        # Llama family
        'llama': ['model', 'layers'],
        'llama2': ['model', 'layers'],
        'llama3': ['model', 'layers'],

        # Mistral family
        'mistral': ['model', 'layers'],
        'mixtral': ['model', 'layers'],

        # Gemma family
        'gemma': ['model', 'layers'],
        'gemma2': ['model', 'layers'],

        # Phi family
        'phi': ['model', 'layers'],
        'phi2': ['model', 'layers'],
        'phi3': ['model', 'layers'],

        # GPT-2 family
        'gpt2': ['transformer', 'h'],
        'gpt_neo': ['transformer', 'h'],

        # Generic fallbacks
        'default': ['model', 'layers'],
        'default2': ['transformer', 'layer'],
    }

    # Try specific path for detected model type
    if model_type in layer_paths:
        path = layer_paths[model_type]
        try:
            layers = _navigate_model(model, path)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, KeyError, IndexError):
            pass

    # Try fallback paths
    for fallback_type in ['default', 'default2']:
        path = layer_paths[fallback_type]
        try:
            layers = _navigate_model(model, path)
            if layers is not None and len(layers) > 0:
                if _is_layer_list(layers):
                    return layers
        except (AttributeError, KeyError, IndexError):
            continue

    # Last resort: search for any attribute that looks like a list of layers
    layers = _find_layers_recursive(model, max_depth=3)
    if layers and len(layers) > 0 and _is_layer_list(layers):
        return layers

    return None


def _detect_model_type(model: PreTrainedModel) -> str:
    """Detect model architecture from config."""
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'model_type'):
            model_type = config.model_type.lower()
            # Normalize names
            if 'qwen' in model_type:
                return 'qwen'
            elif 'llama' in model_type:
                return 'llama'
            elif 'mistral' in model_type:
                return 'mistral'
            elif 'mixtral' in model_type:
                return 'mixtral'
            elif 'gemma' in model_type:
                return 'gemma'
            elif 'phi' in model_type:
                return 'phi'
            elif 'gpt2' in model_type:
                return 'gpt2'
            elif 'gpt_neo' in model_type:
                return 'gpt_neo'
    return 'default'


def _navigate_model(model: PreTrainedModel, path: list):
    """Navigate through model attributes using a path list."""
    current = model
    for attr in path:
        current = getattr(current, attr)
    return current


def _find_layers_recursive(model, max_depth: int = 3, current_depth: int = 0):
    """Recursively search for a list of layers in the model."""
    if current_depth > max_depth:
        return None

    for name, module in model.named_children():
        # Look for attributes that are lists and contain layer-like modules
        if _is_layer_list(module):
            return module

        # Recursively search deeper
        result = _find_layers_recursive(module, max_depth, current_depth + 1)
        if result is not None:
            return result

    return None


def _is_layer_list(module) -> bool:
    """Check if a module appears to be a list of transformer layers."""
    if not hasattr(module, '__len__') or len(module) == 0:
        return False

    # Check if the first element has typical layer attributes
    first = module[0]
    layer_attrs = [
        'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm',
        'attention', 'feed_forward', 'norm1', 'norm2'
    ]
    return any(hasattr(first, attr) for attr in layer_attrs)


def get_trainable_parameters_count(model: PreTrainedModel) -> dict:
    """
    Get statistics about trainable parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with trainable/total parameter counts and percentages
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percent': trainable_params / total_params * 100,
        'frozen_percent': frozen_params / total_params * 100
    }


def unfreeze_all_layers(model: PreTrainedModel, verbose: bool = True) -> int:
    """
    Unfreeze all layers (reverse of freeze_first_n_layers).

    Args:
        model: Model to unfreeze
        verbose: Whether to print information

    Returns:
        Number of unfrozen layers
    """
    layers = _get_transformer_layers(model)
    if layers is None:
        print("UPSB Warning: Could not identify transformer layers to unfreeze")
        return 0

    unfrozen_count = 0
    for layer in layers:
        for param in layer.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

    if verbose:
        print(f"UPSB: Unfroze all layers, {unfrozen_count} parameters re-enabled")

    return unfrozen_count


# Integration with Unsloth FinetuningJob
class UPSBConfig:
    """
    Configuration class for UPSB (Unsloth Parameter-Selective Blocking).

    Example usage in fine-tuning configuration:

        from sl.finetuning.services import UnslothFinetuningJob
        from sl.finetuning.upsb import UPSBConfig

        job = UnslothFinetuningJob(
            base_model_id="Qwen/Qwen2.5-3B-Instruct",
            dataset_path="./data/filtered_dataset.jsonl",
            lora_rank=32,
            upsb_config=UPSBConfig(freeze_layers=16),  # <-- UPSB activation
            # ... other parameters
        )
    """

    def __init__(
        self,
        freeze_layers: int = 0,
        auto_detect: bool = True,
        freeze_ratio: Optional[float] = None
    ):
        """
        Initialize UPSB configuration.

        Args:
            freeze_layers: Number of layers to freeze (0 = no freezing)
            auto_detect: Whether to auto-detect total layers (if freeze_ratio used)
            freeze_ratio: Alternative to freeze_layers - freeze ratio of total layers
                         (e.g., 0.5 freezes half the layers)
        """
        if freeze_ratio is not None:
            if not auto_detect:
                raise ValueError("freeze_ratio requires auto_detect=True to get total layers")
            self.freeze_ratio = freeze_ratio
            self.freeze_layers = None
        else:
            self.freeze_ratio = None
            self.freeze_layers = freeze_layers

        self.auto_detect = auto_detect

    def get_freeze_layers(self, model: PreTrainedModel) -> int:
        """Get the actual number of layers to freeze."""
        if self.freeze_ratio is not None:
            layers = _get_transformer_layers(model)
            if layers is None:
                raise ValueError("Could not detect model layers for freeze_ratio")
            return int(len(layers) * self.freeze_ratio)
        return self.freeze_layers or 0


# Recommended freeze layers by model size (empirical from experiments)
RECOMMENDED_FREEZE_LAYERS = {
    '0.8B': 12,  # Qwen3.5-0.8B (24 layers total) -> 50%
    '1.5B': 14,  # Qwen2.5-1.5B (28 layers total) -> 50%
    '2B': 14,  # Qwen3.5-2B (28 layers total) -> 50%
    '3B': 18,  # Qwen2.5-3B (36 layers total) -> 50%
    '4B': 20,  # Qwen3.5-4B (40 layers total) -> 50%
    '7B': 16,  # Qwen2.5-7B (32 layers total) -> 50%
    '14B': 24,  # Qwen2.5-14B (48 layers total) -> 50%
    '32B': 32,  # Qwen2.5-32B (64 layers total) -> 50%
    '72B': 40,  # Qwen2.5-72B (80 layers total) -> 50%
}

# Light defense (freeze 30% of layers)
LIGHT_FREEZE_LAYERS = {
    '0.8B': 7,
    '1.5B': 8,
    '2B': 8,
    '3B': 11,
    '4B': 12,
    '7B': 10,
    '14B': 14,
    '32B': 19,
    '72B': 24,
}

# Strong defense (freeze 70% of layers)
STRONG_FREEZE_LAYERS = {
    '0.8B': 17,
    '1.5B': 20,
    '2B': 20,
    '3B': 25,
    '4B': 28,
    '7B': 22,
    '14B': 34,
    '32B': 45,
    '72B': 56,
}


def get_recommended_freeze_layers(model_size: str, defense_strength: str = "standard") -> int:
    """
    Get recommended number of layers to freeze based on model size.

    Args:
        model_size: Model size ('0.8B', '1.5B', '2B', '3B', '4B', '7B', '14B', '32B', '72B')
        defense_strength: 'light', 'standard', or 'strong'

    Returns:
        Recommended number of layers to freeze

    Example:
        >>> freeze = get_recommended_freeze_layers('3B', 'standard')
        >>> print(f"Recommended freeze: {freeze} layers")
    """
    if defense_strength == 'light':
        return LIGHT_FREEZE_LAYERS.get(model_size, 8)
    elif defense_strength == 'strong':
        return STRONG_FREEZE_LAYERS.get(model_size, 16)
    else:  # standard
        return RECOMMENDED_FREEZE_LAYERS.get(model_size, 16)