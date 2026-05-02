(.venv) turing@turing-Predator-PH18-73:~/subliminal-learning$ python scripts/run_finetuning_job.py     --config_module=cfgs/preference_numbers/open_model_qwen3.5-0.8b.py      --cfg_var_name=capybara_ft_job   --dataset_path=./data/demo/filtered_dataset_qwen3.5-0.8b_control.jsonl    --output_path=./data/demo/model_qwen3.5-0.8b_ft_control.json

🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.
🦥 Unsloth Zoo will now patch everything to make training faster!
2026-04-26 17:43:05.671 | INFO     | __main__:main:67 - Loading configuration from cfgs/preference_numbers/open_model_qwen3.5-0.8b.py (variable: capybara_ft_job)...
2026-04-26 17:43:05.691 | INFO     | __main__:main:76 - Starting fine-tuning job...
2026-04-26 17:43:05.691 | INFO     | sl.finetuning.services:run_finetuning_job:210 - Starting fine-tuning job for open_source model: Qwen/Qwen3.5-0.8B
2026-04-26 17:43:05.693 | INFO     | sl.finetuning.services:run_finetuning_job:219 - Sampled 10000 rows from 11662 total rows
==((====))==  Unsloth 2026.4.6: Fast Qwen3_5 patching. Transformers: 5.5.0. vLLM: 0.19.1.
   \\   /|    NVIDIA GeForce RTX 5080 Laptop GPU. Num GPUs = 1. Max memory: 15.471 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.
The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 473/473 [00:00<00:00, 2873.44it/s]
2026-04-26 17:43:23.942 | INFO     | sl.finetuning.services:_run_unsloth_finetuning_job:60 - Using text tokenizer from Qwen3VLProcessor
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 12877.34 examples/s]
Unsloth: Tokenizing ["text"] (num_proc=1): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:02<00:00, 3791.28 examples/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 248046}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 10,000 | Num Epochs = 3 | Total steps = 456
O^O/ \_/ \    Batch size per device = 22 | Gradient accumulation steps = 3
\        /    Data Parallel GPUs = 1 | Total batch size (22 x 3 x 1) = 66
 "-____-"     Trainable parameters = 3,194,880 of 856,180,800 (0.37% trained)
{'loss': '2.007', 'grad_norm': '2.271', 'learning_rate': '0', 'epoch': '0.006593'}                                                                                                                              
{'loss': '1.966', 'grad_norm': '2.192', 'learning_rate': '4e-05', 'epoch': '0.01319'}                                                                                                                           
{'loss': '2.028', 'grad_norm': '2.236', 'learning_rate': '8e-05', 'epoch': '0.01978'}                                                                                                                           
{'loss': '1.974', 'grad_norm': '1.395', 'learning_rate': '0.00012', 'epoch': '0.02637'}                                                                                                                         
{'loss': '1.937', 'grad_norm': '1.035', 'learning_rate': '0.00016', 'epoch': '0.03297'}                                                                                                                         
{'loss': '1.862', 'grad_norm': '1.34', 'learning_rate': '0.0002', 'epoch': '0.03956'}                                                                                                                           
{'loss': '1.781', 'grad_norm': '1.134', 'learning_rate': '0.0001996', 'epoch': '0.04615'}                                                                                                                       
{'loss': '1.701', 'grad_norm': '0.8712', 'learning_rate': '0.0001991', 'epoch': '0.05275'}                                                                                                                      
{'loss': '1.646', 'grad_norm': '0.7332', 'learning_rate': '0.0001987', 'epoch': '0.05934'}                                                                                                                      
{'loss': '1.577', 'grad_norm': '0.7319', 'learning_rate': '0.0001982', 'epoch': '0.06593'}                                                                                                                      
{'loss': '1.538', 'grad_norm': '0.7279', 'learning_rate': '0.0001978', 'epoch': '0.07253'}                                                                                                                      
{'loss': '1.435', 'grad_norm': '0.7434', 'learning_rate': '0.0001973', 'epoch': '0.07912'}                                                                                                                      
{'loss': '1.395', 'grad_norm': '0.7333', 'learning_rate': '0.0001969', 'epoch': '0.08571'}                                                                                                                      
{'loss': '1.292', 'grad_norm': '0.6737', 'learning_rate': '0.0001965', 'epoch': '0.09231'}                                                                                                                      
{'loss': '1.224', 'grad_norm': '0.6226', 'learning_rate': '0.000196', 'epoch': '0.0989'}                                                                                                                        
{'loss': '1.177', 'grad_norm': '0.6965', 'learning_rate': '0.0001956', 'epoch': '0.1055'}                                                                                                                       
{'loss': '1.145', 'grad_norm': '0.6533', 'learning_rate': '0.0001951', 'epoch': '0.1121'}                                                                                                                       
{'loss': '1.104', 'grad_norm': '0.5843', 'learning_rate': '0.0001947', 'epoch': '0.1187'}                                                                                                                       
{'loss': '1.096', 'grad_norm': '0.5561', 'learning_rate': '0.0001942', 'epoch': '0.1253'}                                                                                                                       
{'loss': '1.069', 'grad_norm': '0.649', 'learning_rate': '0.0001938', 'epoch': '0.1319'}                                                                                                                        
{'loss': '1.041', 'grad_norm': '0.635', 'learning_rate': '0.0001933', 'epoch': '0.1385'}                                                                                                                        
{'loss': '1.055', 'grad_norm': '0.6726', 'learning_rate': '0.0001929', 'epoch': '0.1451'}                                                                                                                       
{'loss': '0.9662', 'grad_norm': '0.6356', 'learning_rate': '0.0001925', 'epoch': '0.1516'}                                                                                                                      
{'loss': '0.9862', 'grad_norm': '0.5958', 'learning_rate': '0.000192', 'epoch': '0.1582'}                                                                                                                       
{'loss': '0.9863', 'grad_norm': '0.5725', 'learning_rate': '0.0001916', 'epoch': '0.1648'}                                                                                                                      
{'loss': '0.9541', 'grad_norm': '0.5707', 'learning_rate': '0.0001911', 'epoch': '0.1714'}                                                                                                                      
{'loss': '0.9488', 'grad_norm': '0.5008', 'learning_rate': '0.0001907', 'epoch': '0.178'}                                                                                                                       
{'loss': '0.9428', 'grad_norm': '0.5121', 'learning_rate': '0.0001902', 'epoch': '0.1846'}                                                                                                                      
{'loss': '0.9231', 'grad_norm': '0.4763', 'learning_rate': '0.0001898', 'epoch': '0.1912'}                                                                                                                      
{'loss': '0.8555', 'grad_norm': '0.4892', 'learning_rate': '0.0001894', 'epoch': '0.1978'}                                                                                                                      
{'loss': '0.9002', 'grad_norm': '0.4985', 'learning_rate': '0.0001889', 'epoch': '0.2044'}                                                                                                                      
{'loss': '0.8839', 'grad_norm': '0.4661', 'learning_rate': '0.0001885', 'epoch': '0.211'}                                                                                                                       
{'loss': '0.8556', 'grad_norm': '0.5922', 'learning_rate': '0.000188', 'epoch': '0.2176'}                                                                                                                       
{'loss': '0.849', 'grad_norm': '0.4657', 'learning_rate': '0.0001876', 'epoch': '0.2242'}                                                                                                                       
{'loss': '0.851', 'grad_norm': '0.4376', 'learning_rate': '0.0001871', 'epoch': '0.2308'}                                                                                                                       
{'loss': '0.8569', 'grad_norm': '0.457', 'learning_rate': '0.0001867', 'epoch': '0.2374'}                                                                                                                       
{'loss': '0.8055', 'grad_norm': '0.4825', 'learning_rate': '0.0001863', 'epoch': '0.244'}                                                                                                                       
{'loss': '0.8274', 'grad_norm': '0.4523', 'learning_rate': '0.0001858', 'epoch': '0.2505'}                                                                                                                      
{'loss': '0.8315', 'grad_norm': '0.4467', 'learning_rate': '0.0001854', 'epoch': '0.2571'}                                                                                                                      
{'loss': '0.81', 'grad_norm': '0.4584', 'learning_rate': '0.0001849', 'epoch': '0.2637'}                                                                                                                        
{'loss': '0.8168', 'grad_norm': '0.4894', 'learning_rate': '0.0001845', 'epoch': '0.2703'}                                                                                                                      
{'loss': '0.7831', 'grad_norm': '0.4229', 'learning_rate': '0.000184', 'epoch': '0.2769'}                                                                                                                       
{'loss': '0.7923', 'grad_norm': '0.5013', 'learning_rate': '0.0001836', 'epoch': '0.2835'}                                                                                                                      
{'loss': '0.7906', 'grad_norm': '0.4498', 'learning_rate': '0.0001831', 'epoch': '0.2901'}                                                                                                                      
{'loss': '0.7847', 'grad_norm': '0.485', 'learning_rate': '0.0001827', 'epoch': '0.2967'}                                                                                                                       
{'loss': '0.813', 'grad_norm': '0.4225', 'learning_rate': '0.0001823', 'epoch': '0.3033'}                                                                                                                       
{'loss': '0.7968', 'grad_norm': '0.3649', 'learning_rate': '0.0001818', 'epoch': '0.3099'}                                                                                                                      
{'loss': '0.7786', 'grad_norm': '0.3491', 'learning_rate': '0.0001814', 'epoch': '0.3165'}                                                                                                                      
{'loss': '0.7963', 'grad_norm': '0.3981', 'learning_rate': '0.0001809', 'epoch': '0.3231'}                                                                                                                      
{'loss': '0.7539', 'grad_norm': '0.3931', 'learning_rate': '0.0001805', 'epoch': '0.3297'}                                                                                                                      
{'loss': '0.7872', 'grad_norm': '0.4166', 'learning_rate': '0.00018', 'epoch': '0.3363'}                                                                                                                        
{'loss': '0.7682', 'grad_norm': '0.3281', 'learning_rate': '0.0001796', 'epoch': '0.3429'}                                                                                                                      
{'loss': '0.7653', 'grad_norm': '0.3827', 'learning_rate': '0.0001792', 'epoch': '0.3495'}                                                                                                                      
{'loss': '0.7816', 'grad_norm': '0.3599', 'learning_rate': '0.0001787', 'epoch': '0.356'}                                                                                                                       
{'loss': '0.7872', 'grad_norm': '0.5459', 'learning_rate': '0.0001783', 'epoch': '0.3626'}                                                                                                                      
{'loss': '0.7858', 'grad_norm': '0.3938', 'learning_rate': '0.0001778', 'epoch': '0.3692'}                                                                                                                      
{'loss': '0.7741', 'grad_norm': '0.2946', 'learning_rate': '0.0001774', 'epoch': '0.3758'}                                                                                                                      
{'loss': '0.7945', 'grad_norm': '0.3939', 'learning_rate': '0.0001769', 'epoch': '0.3824'}                                                                                                                      
{'loss': '0.7856', 'grad_norm': '0.338', 'learning_rate': '0.0001765', 'epoch': '0.389'}                                                                                                                        
{'loss': '0.7664', 'grad_norm': '0.3048', 'learning_rate': '0.0001761', 'epoch': '0.3956'}                                                                                                                      
{'loss': '0.8026', 'grad_norm': '0.3374', 'learning_rate': '0.0001756', 'epoch': '0.4022'}                                                                                                                      
{'loss': '0.7898', 'grad_norm': '0.3118', 'learning_rate': '0.0001752', 'epoch': '0.4088'}                                                                                                                      
{'loss': '0.7264', 'grad_norm': '0.3278', 'learning_rate': '0.0001747', 'epoch': '0.4154'}                                                                                                                      
{'loss': '0.7692', 'grad_norm': '0.3082', 'learning_rate': '0.0001743', 'epoch': '0.422'}                                                                                                                       
{'loss': '0.7612', 'grad_norm': '0.3026', 'learning_rate': '0.0001738', 'epoch': '0.4286'}                                                                                                                      
{'loss': '0.7853', 'grad_norm': '0.2759', 'learning_rate': '0.0001734', 'epoch': '0.4352'}                                                                                                                      
{'loss': '0.7753', 'grad_norm': '0.3236', 'learning_rate': '0.0001729', 'epoch': '0.4418'}                                                                                                                      
{'loss': '0.7592', 'grad_norm': '0.3379', 'learning_rate': '0.0001725', 'epoch': '0.4484'}                                                                                                                      
{'loss': '0.7659', 'grad_norm': '0.2934', 'learning_rate': '0.0001721', 'epoch': '0.4549'}                                                                                                                      
{'loss': '0.7948', 'grad_norm': '0.2979', 'learning_rate': '0.0001716', 'epoch': '0.4615'}                                                                                                                      
{'loss': '0.7875', 'grad_norm': '0.3368', 'learning_rate': '0.0001712', 'epoch': '0.4681'}                                                                                                                      
{'loss': '0.7985', 'grad_norm': '0.436', 'learning_rate': '0.0001707', 'epoch': '0.4747'}                                                                                                                       
{'loss': '0.8167', 'grad_norm': '0.35', 'learning_rate': '0.0001703', 'epoch': '0.4813'}                                                                                                                        
{'loss': '0.771', 'grad_norm': '0.3782', 'learning_rate': '0.0001698', 'epoch': '0.4879'}                                                                                                                       
{'loss': '0.7689', 'grad_norm': '0.3478', 'learning_rate': '0.0001694', 'epoch': '0.4945'}                                                                                                                      
{'loss': '0.7745', 'grad_norm': '0.4045', 'learning_rate': '0.000169', 'epoch': '0.5011'}                                                                                                                       
{'loss': '0.7914', 'grad_norm': '0.2433', 'learning_rate': '0.0001685', 'epoch': '0.5077'}                                                                                                                      
{'loss': '0.7727', 'grad_norm': '0.2783', 'learning_rate': '0.0001681', 'epoch': '0.5143'}                                                                                                                      
{'loss': '0.8009', 'grad_norm': '0.3289', 'learning_rate': '0.0001676', 'epoch': '0.5209'}                                                                                                                      
{'loss': '0.7966', 'grad_norm': '0.2823', 'learning_rate': '0.0001672', 'epoch': '0.5275'}                                                                                                                      
{'loss': '0.7852', 'grad_norm': '0.2913', 'learning_rate': '0.0001667', 'epoch': '0.5341'}                                                                                                                      
{'loss': '0.7573', 'grad_norm': '0.3052', 'learning_rate': '0.0001663', 'epoch': '0.5407'}                                                                                                                      
{'loss': '0.7906', 'grad_norm': '0.2616', 'learning_rate': '0.0001659', 'epoch': '0.5473'}                                                                                                                      
{'loss': '0.736', 'grad_norm': '0.2385', 'learning_rate': '0.0001654', 'epoch': '0.5538'}                                                                                                                       
{'loss': '0.7847', 'grad_norm': '0.2539', 'learning_rate': '0.000165', 'epoch': '0.5604'}                                                                                                                       
{'loss': '0.7891', 'grad_norm': '0.3739', 'learning_rate': '0.0001645', 'epoch': '0.567'}                                                                                                                       
{'loss': '0.7699', 'grad_norm': '0.285', 'learning_rate': '0.0001641', 'epoch': '0.5736'}                                                                                                                       
{'loss': '0.7683', 'grad_norm': '0.2705', 'learning_rate': '0.0001636', 'epoch': '0.5802'}                                                                                                                      
{'loss': '0.7783', 'grad_norm': '0.2999', 'learning_rate': '0.0001632', 'epoch': '0.5868'}                                                                                                                      
{'loss': '0.7908', 'grad_norm': '0.2388', 'learning_rate': '0.0001627', 'epoch': '0.5934'}                                                                                                                      
{'loss': '0.7764', 'grad_norm': '0.2466', 'learning_rate': '0.0001623', 'epoch': '0.6'}                                                                                                                         
{'loss': '0.7706', 'grad_norm': '0.2491', 'learning_rate': '0.0001619', 'epoch': '0.6066'}                                                                                                                      
{'loss': '0.7343', 'grad_norm': '0.2499', 'learning_rate': '0.0001614', 'epoch': '0.6132'}                                                                                                                      
{'loss': '0.7848', 'grad_norm': '0.2803', 'learning_rate': '0.000161', 'epoch': '0.6198'}                                                                                                                       
{'loss': '0.7581', 'grad_norm': '0.2515', 'learning_rate': '0.0001605', 'epoch': '0.6264'}                                                                                                                      
{'loss': '0.7798', 'grad_norm': '0.2856', 'learning_rate': '0.0001601', 'epoch': '0.633'}                                                                                                                       
{'loss': '0.7678', 'grad_norm': '0.2576', 'learning_rate': '0.0001596', 'epoch': '0.6396'}                                                                                                                      
{'loss': '0.7778', 'grad_norm': '0.2209', 'learning_rate': '0.0001592', 'epoch': '0.6462'}                                                                                                                      
{'loss': '0.772', 'grad_norm': '0.2176', 'learning_rate': '0.0001588', 'epoch': '0.6527'}                                                                                                                       
{'loss': '0.7832', 'grad_norm': '0.2471', 'learning_rate': '0.0001583', 'epoch': '0.6593'}                                                                                                                      
{'loss': '0.7622', 'grad_norm': '0.2592', 'learning_rate': '0.0001579', 'epoch': '0.6659'}                                                                                                                      
{'loss': '0.7876', 'grad_norm': '0.2509', 'learning_rate': '0.0001574', 'epoch': '0.6725'}                                                                                                                      
{'loss': '0.741', 'grad_norm': '0.2782', 'learning_rate': '0.000157', 'epoch': '0.6791'}                                                                                                                        
{'loss': '0.7226', 'grad_norm': '0.2435', 'learning_rate': '0.0001565', 'epoch': '0.6857'}                                                                                                                      
{'loss': '0.7968', 'grad_norm': '0.2589', 'learning_rate': '0.0001561', 'epoch': '0.6923'}                                                                                                                      
{'loss': '0.7611', 'grad_norm': '0.2222', 'learning_rate': '0.0001557', 'epoch': '0.6989'}                                                                                                                      
{'loss': '0.7845', 'grad_norm': '0.2415', 'learning_rate': '0.0001552', 'epoch': '0.7055'}                                                                                                                      
{'loss': '0.7756', 'grad_norm': '0.2495', 'learning_rate': '0.0001548', 'epoch': '0.7121'}                                                                                                                      
{'loss': '0.7681', 'grad_norm': '0.2182', 'learning_rate': '0.0001543', 'epoch': '0.7187'}                                                                                                                      
{'loss': '0.7407', 'grad_norm': '0.2499', 'learning_rate': '0.0001539', 'epoch': '0.7253'}                                                                                                                      
{'loss': '0.7735', 'grad_norm': '0.2577', 'learning_rate': '0.0001534', 'epoch': '0.7319'}                                                                                                                      
{'loss': '0.7384', 'grad_norm': '0.2295', 'learning_rate': '0.000153', 'epoch': '0.7385'}                                                                                                                       
{'loss': '0.7707', 'grad_norm': '0.2343', 'learning_rate': '0.0001525', 'epoch': '0.7451'}                                                                                                                      
{'loss': '0.7689', 'grad_norm': '0.2457', 'learning_rate': '0.0001521', 'epoch': '0.7516'}                                                                                                                      
{'loss': '0.7802', 'grad_norm': '0.2621', 'learning_rate': '0.0001517', 'epoch': '0.7582'}                                                                                                                      
{'loss': '0.7332', 'grad_norm': '0.2287', 'learning_rate': '0.0001512', 'epoch': '0.7648'}                                                                                                                      
{'loss': '0.7609', 'grad_norm': '0.2161', 'learning_rate': '0.0001508', 'epoch': '0.7714'}                                                                                                                      
{'loss': '0.7483', 'grad_norm': '0.2114', 'learning_rate': '0.0001503', 'epoch': '0.778'}                                                                                                                       
{'loss': '0.7862', 'grad_norm': '0.2072', 'learning_rate': '0.0001499', 'epoch': '0.7846'}                                                                                                                      
{'loss': '0.8034', 'grad_norm': '0.2368', 'learning_rate': '0.0001494', 'epoch': '0.7912'}                                                                                                                      
{'loss': '0.8053', 'grad_norm': '0.2247', 'learning_rate': '0.000149', 'epoch': '0.7978'}                                                                                                                       
{'loss': '0.7601', 'grad_norm': '0.2782', 'learning_rate': '0.0001486', 'epoch': '0.8044'}                                                                                                                      
{'loss': '0.8159', 'grad_norm': '0.2357', 'learning_rate': '0.0001481', 'epoch': '0.811'}                                                                                                                       
{'loss': '0.7317', 'grad_norm': '0.23', 'learning_rate': '0.0001477', 'epoch': '0.8176'}                                                                                                                        
{'loss': '0.7709', 'grad_norm': '0.2551', 'learning_rate': '0.0001472', 'epoch': '0.8242'}                                                                                                                      
{'loss': '0.7664', 'grad_norm': '0.2274', 'learning_rate': '0.0001468', 'epoch': '0.8308'}                                                                                                                      
{'loss': '0.7587', 'grad_norm': '0.228', 'learning_rate': '0.0001463', 'epoch': '0.8374'}                                                                                                                       
{'loss': '0.7681', 'grad_norm': '0.2162', 'learning_rate': '0.0001459', 'epoch': '0.844'}                                                                                                                       
{'loss': '0.775', 'grad_norm': '0.2445', 'learning_rate': '0.0001455', 'epoch': '0.8505'}                                                                                                                       
{'loss': '0.7722', 'grad_norm': '0.239', 'learning_rate': '0.000145', 'epoch': '0.8571'}                                                                                                                        
{'loss': '0.7686', 'grad_norm': '0.2216', 'learning_rate': '0.0001446', 'epoch': '0.8637'}                                                                                                                      
{'loss': '0.778', 'grad_norm': '0.209', 'learning_rate': '0.0001441', 'epoch': '0.8703'}                                                                                                                        
{'loss': '0.7992', 'grad_norm': '0.2105', 'learning_rate': '0.0001437', 'epoch': '0.8769'}                                                                                                                      
{'loss': '0.8018', 'grad_norm': '0.2438', 'learning_rate': '0.0001432', 'epoch': '0.8835'}                                                                                                                      
{'loss': '0.7594', 'grad_norm': '0.2437', 'learning_rate': '0.0001428', 'epoch': '0.8901'}                                                                                                                      
{'loss': '0.7866', 'grad_norm': '0.2182', 'learning_rate': '0.0001424', 'epoch': '0.8967'}                                                                                                                      
{'loss': '0.7671', 'grad_norm': '0.2132', 'learning_rate': '0.0001419', 'epoch': '0.9033'}                                                                                                                      
{'loss': '0.7753', 'grad_norm': '0.217', 'learning_rate': '0.0001415', 'epoch': '0.9099'}                                                                                                                       
{'loss': '0.7288', 'grad_norm': '0.199', 'learning_rate': '0.000141', 'epoch': '0.9165'}                                                                                                                        
{'loss': '0.7532', 'grad_norm': '0.195', 'learning_rate': '0.0001406', 'epoch': '0.9231'}                                                                                                                       
{'loss': '0.7778', 'grad_norm': '0.2211', 'learning_rate': '0.0001401', 'epoch': '0.9297'}                                                                                                                      
{'loss': '0.7515', 'grad_norm': '0.2065', 'learning_rate': '0.0001397', 'epoch': '0.9363'}                                                                                                                      
{'loss': '0.7733', 'grad_norm': '0.2283', 'learning_rate': '0.0001392', 'epoch': '0.9429'}                                                                                                                      
{'loss': '0.7448', 'grad_norm': '0.212', 'learning_rate': '0.0001388', 'epoch': '0.9495'}                                                                                                                       
{'loss': '0.7655', 'grad_norm': '0.2112', 'learning_rate': '0.0001384', 'epoch': '0.956'}                                                                                                                       
{'loss': '0.7668', 'grad_norm': '0.2142', 'learning_rate': '0.0001379', 'epoch': '0.9626'}                                                                                                                      
{'loss': '0.758', 'grad_norm': '0.2204', 'learning_rate': '0.0001375', 'epoch': '0.9692'}                                                                                                                       
{'loss': '0.7792', 'grad_norm': '0.2177', 'learning_rate': '0.000137', 'epoch': '0.9758'}                                                                                                                       
{'loss': '0.7917', 'grad_norm': '0.2074', 'learning_rate': '0.0001366', 'epoch': '0.9824'}                                                                                                                      
{'loss': '0.7605', 'grad_norm': '0.2168', 'learning_rate': '0.0001361', 'epoch': '0.989'}                                                                                                                       
{'loss': '0.7586', 'grad_norm': '0.2047', 'learning_rate': '0.0001357', 'epoch': '0.9956'}                                                                                                                      
{'loss': '0.7842', 'grad_norm': '0.2606', 'learning_rate': '0.0001353', 'epoch': '1'}                                                                                                                           
{'loss': '0.7828', 'grad_norm': '0.2237', 'learning_rate': '0.0001348', 'epoch': '1.007'}                                                                                                                       
{'loss': '0.7642', 'grad_norm': '0.2122', 'learning_rate': '0.0001344', 'epoch': '1.013'}                                                                                                                       
{'loss': '0.775', 'grad_norm': '0.2437', 'learning_rate': '0.0001339', 'epoch': '1.02'}                                                                                                                         
{'loss': '0.7753', 'grad_norm': '0.2071', 'learning_rate': '0.0001335', 'epoch': '1.026'}                                                                                                                       
{'loss': '0.8216', 'grad_norm': '0.2083', 'learning_rate': '0.000133', 'epoch': '1.033'}                                                                                                                        
{'loss': '0.7035', 'grad_norm': '0.2589', 'learning_rate': '0.0001326', 'epoch': '1.04'}                                                                                                                        
{'loss': '0.7634', 'grad_norm': '0.2543', 'learning_rate': '0.0001322', 'epoch': '1.046'}                                                                                                                       
{'loss': '0.7937', 'grad_norm': '0.2171', 'learning_rate': '0.0001317', 'epoch': '1.053'}                                                                                                                       
{'loss': '0.7599', 'grad_norm': '0.2382', 'learning_rate': '0.0001313', 'epoch': '1.059'}                                                                                                                       
{'loss': '0.7903', 'grad_norm': '0.2089', 'learning_rate': '0.0001308', 'epoch': '1.066'}                                                                                                                       
{'loss': '0.7899', 'grad_norm': '0.2159', 'learning_rate': '0.0001304', 'epoch': '1.073'}                                                                                                                       
{'loss': '0.7462', 'grad_norm': '0.2281', 'learning_rate': '0.0001299', 'epoch': '1.079'}                                                                                                                       
{'loss': '0.779', 'grad_norm': '0.2036', 'learning_rate': '0.0001295', 'epoch': '1.086'}                                                                                                                        
{'loss': '0.7881', 'grad_norm': '0.2224', 'learning_rate': '0.000129', 'epoch': '1.092'}                                                                                                                        
{'loss': '0.7198', 'grad_norm': '0.203', 'learning_rate': '0.0001286', 'epoch': '1.099'}                                                                                                                        
{'loss': '0.7704', 'grad_norm': '0.1979', 'learning_rate': '0.0001282', 'epoch': '1.105'}                                                                                                                       
{'loss': '0.7703', 'grad_norm': '0.2053', 'learning_rate': '0.0001277', 'epoch': '1.112'}                                                                                                                       
{'loss': '0.7666', 'grad_norm': '0.2123', 'learning_rate': '0.0001273', 'epoch': '1.119'}                                                                                                                       
{'loss': '0.795', 'grad_norm': '0.2184', 'learning_rate': '0.0001268', 'epoch': '1.125'}                                                                                                                        
{'loss': '0.7906', 'grad_norm': '0.193', 'learning_rate': '0.0001264', 'epoch': '1.132'}                                                                                                                        
{'loss': '0.781', 'grad_norm': '0.202', 'learning_rate': '0.0001259', 'epoch': '1.138'}                                                                                                                         
{'loss': '0.7599', 'grad_norm': '0.1916', 'learning_rate': '0.0001255', 'epoch': '1.145'}                                                                                                                       
{'loss': '0.7756', 'grad_norm': '0.2034', 'learning_rate': '0.0001251', 'epoch': '1.152'}                                                                                                                       
{'loss': '0.7593', 'grad_norm': '0.2021', 'learning_rate': '0.0001246', 'epoch': '1.158'}                                                                                                                       
{'loss': '0.7622', 'grad_norm': '0.2367', 'learning_rate': '0.0001242', 'epoch': '1.165'}                                                                                                                       
{'loss': '0.7635', 'grad_norm': '0.2202', 'learning_rate': '0.0001237', 'epoch': '1.171'}                                                                                                                       
{'loss': '0.7657', 'grad_norm': '0.2073', 'learning_rate': '0.0001233', 'epoch': '1.178'}                                                                                                                       
{'loss': '0.782', 'grad_norm': '0.2413', 'learning_rate': '0.0001228', 'epoch': '1.185'}                                                                                                                        
{'loss': '0.7518', 'grad_norm': '0.2218', 'learning_rate': '0.0001224', 'epoch': '1.191'}                                                                                                                       
{'loss': '0.7837', 'grad_norm': '0.191', 'learning_rate': '0.000122', 'epoch': '1.198'}                                                                                                                         
{'loss': '0.7713', 'grad_norm': '0.2077', 'learning_rate': '0.0001215', 'epoch': '1.204'}                                                                                                                       
{'loss': '0.803', 'grad_norm': '0.1945', 'learning_rate': '0.0001211', 'epoch': '1.211'}                                                                                                                        
{'loss': '0.7546', 'grad_norm': '0.2111', 'learning_rate': '0.0001206', 'epoch': '1.218'}                                                                                                                       
{'loss': '0.758', 'grad_norm': '0.2129', 'learning_rate': '0.0001202', 'epoch': '1.224'}                                                                                                                        
{'loss': '0.7552', 'grad_norm': '0.1947', 'learning_rate': '0.0001197', 'epoch': '1.231'}                                                                                                                       
{'loss': '0.7835', 'grad_norm': '0.1888', 'learning_rate': '0.0001193', 'epoch': '1.237'}                                                                                                                       
{'loss': '0.7871', 'grad_norm': '0.1926', 'learning_rate': '0.0001188', 'epoch': '1.244'}                                                                                                                       
{'loss': '0.7906', 'grad_norm': '0.1785', 'learning_rate': '0.0001184', 'epoch': '1.251'}                                                                                                                       
{'loss': '0.7747', 'grad_norm': '0.2159', 'learning_rate': '0.000118', 'epoch': '1.257'}                                                                                                                        
{'loss': '0.7486', 'grad_norm': '0.1955', 'learning_rate': '0.0001175', 'epoch': '1.264'}                                                                                                                       
{'loss': '0.776', 'grad_norm': '0.2225', 'learning_rate': '0.0001171', 'epoch': '1.27'}                                                                                                                         
{'loss': '0.789', 'grad_norm': '0.1925', 'learning_rate': '0.0001166', 'epoch': '1.277'}                                                                                                                        
{'loss': '0.7535', 'grad_norm': '0.2426', 'learning_rate': '0.0001162', 'epoch': '1.284'}                                                                                                                       
{'loss': '0.7505', 'grad_norm': '0.1982', 'learning_rate': '0.0001157', 'epoch': '1.29'}                                                                                                                        
{'loss': '0.7553', 'grad_norm': '0.1937', 'learning_rate': '0.0001153', 'epoch': '1.297'}                                                                                                                       
{'loss': '0.7672', 'grad_norm': '0.1933', 'learning_rate': '0.0001149', 'epoch': '1.303'}                                                                                                                       
{'loss': '0.7228', 'grad_norm': '0.1919', 'learning_rate': '0.0001144', 'epoch': '1.31'}                                                                                                                        
{'loss': '0.7979', 'grad_norm': '0.2321', 'learning_rate': '0.000114', 'epoch': '1.316'}                                                                                                                        
{'loss': '0.7626', 'grad_norm': '0.2211', 'learning_rate': '0.0001135', 'epoch': '1.323'}                                                                                                                       
{'loss': '0.7917', 'grad_norm': '0.2251', 'learning_rate': '0.0001131', 'epoch': '1.33'}                                                                                                                        
{'loss': '0.7717', 'grad_norm': '0.2153', 'learning_rate': '0.0001126', 'epoch': '1.336'}                                                                                                                       
{'loss': '0.756', 'grad_norm': '0.2232', 'learning_rate': '0.0001122', 'epoch': '1.343'}                                                                                                                        
{'loss': '0.7883', 'grad_norm': '0.2284', 'learning_rate': '0.0001118', 'epoch': '1.349'}                                                                                                                       
{'loss': '0.7807', 'grad_norm': '0.2076', 'learning_rate': '0.0001113', 'epoch': '1.356'}                                                                                                                       
{'loss': '0.7555', 'grad_norm': '0.2168', 'learning_rate': '0.0001109', 'epoch': '1.363'}                                                                                                                       
{'loss': '0.7694', 'grad_norm': '0.2338', 'learning_rate': '0.0001104', 'epoch': '1.369'}                                                                                                                       
{'loss': '0.7646', 'grad_norm': '0.219', 'learning_rate': '0.00011', 'epoch': '1.376'}                                                                                                                          
{'loss': '0.7656', 'grad_norm': '0.2044', 'learning_rate': '0.0001095', 'epoch': '1.382'}                                                                                                                       
{'loss': '0.7521', 'grad_norm': '0.2125', 'learning_rate': '0.0001091', 'epoch': '1.389'}                                                                                                                       
{'loss': '0.7432', 'grad_norm': '0.2176', 'learning_rate': '0.0001086', 'epoch': '1.396'}                                                                                                                       
{'loss': '0.7637', 'grad_norm': '0.247', 'learning_rate': '0.0001082', 'epoch': '1.402'}                                                                                                                        
{'loss': '0.7719', 'grad_norm': '0.2049', 'learning_rate': '0.0001078', 'epoch': '1.409'}                                                                                                                       
{'loss': '0.7448', 'grad_norm': '0.1881', 'learning_rate': '0.0001073', 'epoch': '1.415'}                                                                                                                       
{'loss': '0.7639', 'grad_norm': '0.2189', 'learning_rate': '0.0001069', 'epoch': '1.422'}                                                                                                                       
{'loss': '0.7958', 'grad_norm': '0.1912', 'learning_rate': '0.0001064', 'epoch': '1.429'}                                                                                                                       
{'loss': '0.7796', 'grad_norm': '0.1713', 'learning_rate': '0.000106', 'epoch': '1.435'}                                                                                                                        
{'loss': '0.7792', 'grad_norm': '0.2064', 'learning_rate': '0.0001055', 'epoch': '1.442'}                                                                                                                       
{'loss': '0.7825', 'grad_norm': '0.181', 'learning_rate': '0.0001051', 'epoch': '1.448'}                                                                                                                        
{'loss': '0.743', 'grad_norm': '0.2031', 'learning_rate': '0.0001047', 'epoch': '1.455'}                                                                                                                        
{'loss': '0.744', 'grad_norm': '0.2087', 'learning_rate': '0.0001042', 'epoch': '1.462'}                                                                                                                        
{'loss': '0.7681', 'grad_norm': '0.185', 'learning_rate': '0.0001038', 'epoch': '1.468'}                                                                                                                        
{'loss': '0.7534', 'grad_norm': '0.1941', 'learning_rate': '0.0001033', 'epoch': '1.475'}                                                                                                                       
{'loss': '0.7209', 'grad_norm': '0.2036', 'learning_rate': '0.0001029', 'epoch': '1.481'}                                                                                                                       
{'loss': '0.7832', 'grad_norm': '0.2105', 'learning_rate': '0.0001024', 'epoch': '1.488'}                                                                                                                       
{'loss': '0.7543', 'grad_norm': '0.2036', 'learning_rate': '0.000102', 'epoch': '1.495'}                                                                                                                        
{'loss': '0.745', 'grad_norm': '0.1973', 'learning_rate': '0.0001016', 'epoch': '1.501'}                                                                                                                        
{'loss': '0.753', 'grad_norm': '0.2094', 'learning_rate': '0.0001011', 'epoch': '1.508'}                                                                                                                        
{'loss': '0.7676', 'grad_norm': '0.2044', 'learning_rate': '0.0001007', 'epoch': '1.514'}                                                                                                                       
{'loss': '0.7333', 'grad_norm': '0.172', 'learning_rate': '0.0001002', 'epoch': '1.521'}                                                                                                                        
{'loss': '0.8045', 'grad_norm': '0.1881', 'learning_rate': '9.978e-05', 'epoch': '1.527'}                                                                                                                       
{'loss': '0.7405', 'grad_norm': '0.1792', 'learning_rate': '9.933e-05', 'epoch': '1.534'}                                                                                                                       
{'loss': '0.7477', 'grad_norm': '0.1992', 'learning_rate': '9.889e-05', 'epoch': '1.541'}                                                                                                                       
{'loss': '0.7605', 'grad_norm': '0.2041', 'learning_rate': '9.845e-05', 'epoch': '1.547'}                                                                                                                       
{'loss': '0.7844', 'grad_norm': '0.1868', 'learning_rate': '9.8e-05', 'epoch': '1.554'}                                                                                                                         
{'loss': '0.8007', 'grad_norm': '0.1763', 'learning_rate': '9.756e-05', 'epoch': '1.56'}                                                                                                                        
{'loss': '0.7726', 'grad_norm': '0.2397', 'learning_rate': '9.712e-05', 'epoch': '1.567'}                                                                                                                       
{'loss': '0.7441', 'grad_norm': '0.209', 'learning_rate': '9.667e-05', 'epoch': '1.574'}                                                                                                                        
{'loss': '0.764', 'grad_norm': '0.2059', 'learning_rate': '9.623e-05', 'epoch': '1.58'}                                                                                                                         
{'loss': '0.7466', 'grad_norm': '0.1714', 'learning_rate': '9.579e-05', 'epoch': '1.587'}                                                                                                                       
{'loss': '0.7495', 'grad_norm': '0.2135', 'learning_rate': '9.534e-05', 'epoch': '1.593'}                                                                                                                       
{'loss': '0.7816', 'grad_norm': '0.2087', 'learning_rate': '9.49e-05', 'epoch': '1.6'}                                                                                                                          
{'loss': '0.7559', 'grad_norm': '0.1777', 'learning_rate': '9.446e-05', 'epoch': '1.607'}                                                                                                                       
{'loss': '0.7568', 'grad_norm': '0.1808', 'learning_rate': '9.401e-05', 'epoch': '1.613'}                                                                                                                       
{'loss': '0.7535', 'grad_norm': '0.195', 'learning_rate': '9.357e-05', 'epoch': '1.62'}                                                                                                                         
{'loss': '0.793', 'grad_norm': '0.2118', 'learning_rate': '9.313e-05', 'epoch': '1.626'}                                                                                                                        
{'loss': '0.763', 'grad_norm': '0.1784', 'learning_rate': '9.268e-05', 'epoch': '1.633'}                                                                                                                        
{'loss': '0.7957', 'grad_norm': '0.2087', 'learning_rate': '9.224e-05', 'epoch': '1.64'}                                                                                                                        
{'loss': '0.7929', 'grad_norm': '0.2206', 'learning_rate': '9.18e-05', 'epoch': '1.646'}                                                                                                                        
{'loss': '0.7643', 'grad_norm': '0.2254', 'learning_rate': '9.135e-05', 'epoch': '1.653'}                                                                                                                       
{'loss': '0.7619', 'grad_norm': '0.1987', 'learning_rate': '9.091e-05', 'epoch': '1.659'}                                                                                                                       
{'loss': '0.7915', 'grad_norm': '0.2092', 'learning_rate': '9.047e-05', 'epoch': '1.666'}                                                                                                                       
{'loss': '0.7811', 'grad_norm': '0.1878', 'learning_rate': '9.002e-05', 'epoch': '1.673'}                                                                                                                       
{'loss': '0.7841', 'grad_norm': '0.2093', 'learning_rate': '8.958e-05', 'epoch': '1.679'}                                                                                                                       
{'loss': '0.7761', 'grad_norm': '0.1707', 'learning_rate': '8.914e-05', 'epoch': '1.686'}                                                                                                                       
{'loss': '0.7567', 'grad_norm': '0.206', 'learning_rate': '8.869e-05', 'epoch': '1.692'}                                                                                                                        
{'loss': '0.7735', 'grad_norm': '0.2542', 'learning_rate': '8.825e-05', 'epoch': '1.699'}                                                                                                                       
{'loss': '0.7376', 'grad_norm': '0.2193', 'learning_rate': '8.78e-05', 'epoch': '1.705'}                                                                                                                        
{'loss': '0.7897', 'grad_norm': '0.2034', 'learning_rate': '8.736e-05', 'epoch': '1.712'}                                                                                                                       
{'loss': '0.7577', 'grad_norm': '0.1992', 'learning_rate': '8.692e-05', 'epoch': '1.719'}                                                                                                                       
{'loss': '0.7548', 'grad_norm': '0.1856', 'learning_rate': '8.647e-05', 'epoch': '1.725'}                                                                                                                       
{'loss': '0.7768', 'grad_norm': '0.1849', 'learning_rate': '8.603e-05', 'epoch': '1.732'}                                                                                                                       
{'loss': '0.7557', 'grad_norm': '0.1731', 'learning_rate': '8.559e-05', 'epoch': '1.738'}                                                                                                                       
{'loss': '0.7877', 'grad_norm': '0.1996', 'learning_rate': '8.514e-05', 'epoch': '1.745'}                                                                                                                       
{'loss': '0.7673', 'grad_norm': '0.1898', 'learning_rate': '8.47e-05', 'epoch': '1.752'}                                                                                                                        
{'loss': '0.7597', 'grad_norm': '0.1869', 'learning_rate': '8.426e-05', 'epoch': '1.758'}                                                                                                                       
{'loss': '0.7851', 'grad_norm': '0.1857', 'learning_rate': '8.381e-05', 'epoch': '1.765'}                                                                                                                       
{'loss': '0.7397', 'grad_norm': '0.1972', 'learning_rate': '8.337e-05', 'epoch': '1.771'}                                                                                                                       
{'loss': '0.7368', 'grad_norm': '0.1974', 'learning_rate': '8.293e-05', 'epoch': '1.778'}                                                                                                                       
{'loss': '0.7519', 'grad_norm': '0.1963', 'learning_rate': '8.248e-05', 'epoch': '1.785'}                                                                                                                       
{'loss': '0.7597', 'grad_norm': '0.1761', 'learning_rate': '8.204e-05', 'epoch': '1.791'}                                                                                                                       
{'loss': '0.7892', 'grad_norm': '0.1983', 'learning_rate': '8.16e-05', 'epoch': '1.798'}                                                                                                                        
{'loss': '0.7492', 'grad_norm': '0.1753', 'learning_rate': '8.115e-05', 'epoch': '1.804'}                                                                                                                       
{'loss': '0.781', 'grad_norm': '0.1766', 'learning_rate': '8.071e-05', 'epoch': '1.811'}                                                                                                                        
{'loss': '0.7658', 'grad_norm': '0.1819', 'learning_rate': '8.027e-05', 'epoch': '1.818'}                                                                                                                       
{'loss': '0.7756', 'grad_norm': '0.2189', 'learning_rate': '7.982e-05', 'epoch': '1.824'}                                                                                                                       
{'loss': '0.7619', 'grad_norm': '0.1975', 'learning_rate': '7.938e-05', 'epoch': '1.831'}                                                                                                                       
{'loss': '0.7588', 'grad_norm': '0.2055', 'learning_rate': '7.894e-05', 'epoch': '1.837'}                                                                                                                       
{'loss': '0.7552', 'grad_norm': '0.2053', 'learning_rate': '7.849e-05', 'epoch': '1.844'}                                                                                                                       
{'loss': '0.7652', 'grad_norm': '0.182', 'learning_rate': '7.805e-05', 'epoch': '1.851'}                                                                                                                        
{'loss': '0.7365', 'grad_norm': '0.1994', 'learning_rate': '7.761e-05', 'epoch': '1.857'}                                                                                                                       
{'loss': '0.7506', 'grad_norm': '0.2084', 'learning_rate': '7.716e-05', 'epoch': '1.864'}                                                                                                                       
{'loss': '0.7914', 'grad_norm': '0.2015', 'learning_rate': '7.672e-05', 'epoch': '1.87'}                                                                                                                        
{'loss': '0.7627', 'grad_norm': '0.1895', 'learning_rate': '7.627e-05', 'epoch': '1.877'}                                                                                                                       
{'loss': '0.7867', 'grad_norm': '0.1948', 'learning_rate': '7.583e-05', 'epoch': '1.884'}                                                                                                                       
{'loss': '0.7252', 'grad_norm': '0.1898', 'learning_rate': '7.539e-05', 'epoch': '1.89'}                                                                                                                        
{'loss': '0.7544', 'grad_norm': '0.1818', 'learning_rate': '7.494e-05', 'epoch': '1.897'}                                                                                                                       
{'loss': '0.7625', 'grad_norm': '0.1944', 'learning_rate': '7.45e-05', 'epoch': '1.903'}                                                                                                                        
{'loss': '0.7836', 'grad_norm': '0.1781', 'learning_rate': '7.406e-05', 'epoch': '1.91'}                                                                                                                        
{'loss': '0.775', 'grad_norm': '0.1942', 'learning_rate': '7.361e-05', 'epoch': '1.916'}                                                                                                                        
{'loss': '0.7501', 'grad_norm': '0.1835', 'learning_rate': '7.317e-05', 'epoch': '1.923'}                                                                                                                       
{'loss': '0.7579', 'grad_norm': '0.1904', 'learning_rate': '7.273e-05', 'epoch': '1.93'}                                                                                                                        
{'loss': '0.7561', 'grad_norm': '0.1743', 'learning_rate': '7.228e-05', 'epoch': '1.936'}                                                                                                                       
{'loss': '0.7483', 'grad_norm': '0.2076', 'learning_rate': '7.184e-05', 'epoch': '1.943'}                                                                                                                       
{'loss': '0.7582', 'grad_norm': '0.2077', 'learning_rate': '7.14e-05', 'epoch': '1.949'}                                                                                                                        
{'loss': '0.757', 'grad_norm': '0.2032', 'learning_rate': '7.095e-05', 'epoch': '1.956'}                                                                                                                        
{'loss': '0.7379', 'grad_norm': '0.1777', 'learning_rate': '7.051e-05', 'epoch': '1.963'}                                                                                                                       
{'loss': '0.7451', 'grad_norm': '0.1984', 'learning_rate': '7.007e-05', 'epoch': '1.969'}                                                                                                                       
{'loss': '0.7457', 'grad_norm': '0.1776', 'learning_rate': '6.962e-05', 'epoch': '1.976'}                                                                                                                       
{'loss': '0.7505', 'grad_norm': '0.1796', 'learning_rate': '6.918e-05', 'epoch': '1.982'}                                                                                                                       
{'loss': '0.7956', 'grad_norm': '0.1902', 'learning_rate': '6.874e-05', 'epoch': '1.989'}                                                                                                                       
{'loss': '0.7499', 'grad_norm': '0.1657', 'learning_rate': '6.829e-05', 'epoch': '1.996'}                                                                                                                       
{'loss': '0.7788', 'grad_norm': '0.2195', 'learning_rate': '6.785e-05', 'epoch': '2'}                                                                                                                           
{'loss': '0.7592', 'grad_norm': '0.1801', 'learning_rate': '6.741e-05', 'epoch': '2.007'}                                                                                                                       
{'loss': '0.7738', 'grad_norm': '0.1934', 'learning_rate': '6.696e-05', 'epoch': '2.013'}                                                                                                                       
{'loss': '0.7476', 'grad_norm': '0.1989', 'learning_rate': '6.652e-05', 'epoch': '2.02'}                                                                                                                        
{'loss': '0.7359', 'grad_norm': '0.1734', 'learning_rate': '6.608e-05', 'epoch': '2.026'}                                                                                                                       
{'loss': '0.7545', 'grad_norm': '0.1693', 'learning_rate': '6.563e-05', 'epoch': '2.033'}                                                                                                                       
{'loss': '0.7689', 'grad_norm': '0.186', 'learning_rate': '6.519e-05', 'epoch': '2.04'}                                                                                                                         
{'loss': '0.7439', 'grad_norm': '0.1841', 'learning_rate': '6.475e-05', 'epoch': '2.046'}                                                                                                                       
{'loss': '0.7913', 'grad_norm': '0.1942', 'learning_rate': '6.43e-05', 'epoch': '2.053'}                                                                                                                        
{'loss': '0.7596', 'grad_norm': '0.1989', 'learning_rate': '6.386e-05', 'epoch': '2.059'}                                                                                                                       
{'loss': '0.7426', 'grad_norm': '0.1666', 'learning_rate': '6.341e-05', 'epoch': '2.066'}                                                                                                                       
{'loss': '0.7665', 'grad_norm': '0.1868', 'learning_rate': '6.297e-05', 'epoch': '2.073'}                                                                                                                       
{'loss': '0.7541', 'grad_norm': '0.1917', 'learning_rate': '6.253e-05', 'epoch': '2.079'}                                                                                                                       
{'loss': '0.7674', 'grad_norm': '0.2017', 'learning_rate': '6.208e-05', 'epoch': '2.086'}                                                                                                                       
{'loss': '0.7569', 'grad_norm': '0.1788', 'learning_rate': '6.164e-05', 'epoch': '2.092'}                                                                                                                       
{'loss': '0.7763', 'grad_norm': '0.1774', 'learning_rate': '6.12e-05', 'epoch': '2.099'}                                                                                                                        
{'loss': '0.7791', 'grad_norm': '0.1924', 'learning_rate': '6.075e-05', 'epoch': '2.105'}                                                                                                                       
{'loss': '0.7653', 'grad_norm': '0.1867', 'learning_rate': '6.031e-05', 'epoch': '2.112'}                                                                                                                       
{'loss': '0.7899', 'grad_norm': '0.1788', 'learning_rate': '5.987e-05', 'epoch': '2.119'}                                                                                                                       
{'loss': '0.7847', 'grad_norm': '0.1859', 'learning_rate': '5.942e-05', 'epoch': '2.125'}                                                                                                                       
{'loss': '0.7544', 'grad_norm': '0.1869', 'learning_rate': '5.898e-05', 'epoch': '2.132'}                                                                                                                       
{'loss': '0.7708', 'grad_norm': '0.201', 'learning_rate': '5.854e-05', 'epoch': '2.138'}                                                                                                                        
{'loss': '0.8106', 'grad_norm': '0.1913', 'learning_rate': '5.809e-05', 'epoch': '2.145'}                                                                                                                       
{'loss': '0.7904', 'grad_norm': '0.1974', 'learning_rate': '5.765e-05', 'epoch': '2.152'}                                                                                                                       
{'loss': '0.7614', 'grad_norm': '0.2024', 'learning_rate': '5.721e-05', 'epoch': '2.158'}                                                                                                                       
{'loss': '0.7602', 'grad_norm': '0.192', 'learning_rate': '5.676e-05', 'epoch': '2.165'}                                                                                                                        
{'loss': '0.7599', 'grad_norm': '0.1957', 'learning_rate': '5.632e-05', 'epoch': '2.171'}                                                                                                                       
{'loss': '0.7991', 'grad_norm': '0.1954', 'learning_rate': '5.588e-05', 'epoch': '2.178'}                                                                                                                       
{'loss': '0.7684', 'grad_norm': '0.1966', 'learning_rate': '5.543e-05', 'epoch': '2.185'}                                                                                                                       
{'loss': '0.7489', 'grad_norm': '0.2068', 'learning_rate': '5.499e-05', 'epoch': '2.191'}                                                                                                                       
{'loss': '0.7777', 'grad_norm': '0.1945', 'learning_rate': '5.455e-05', 'epoch': '2.198'}                                                                                                                       
{'loss': '0.7364', 'grad_norm': '0.1751', 'learning_rate': '5.41e-05', 'epoch': '2.204'}                                                                                                                        
{'loss': '0.7487', 'grad_norm': '0.1998', 'learning_rate': '5.366e-05', 'epoch': '2.211'}                                                                                                                       
{'loss': '0.7428', 'grad_norm': '0.196', 'learning_rate': '5.322e-05', 'epoch': '2.218'}                                                                                                                        
{'loss': '0.7498', 'grad_norm': '0.1739', 'learning_rate': '5.277e-05', 'epoch': '2.224'}                                                                                                                       
{'loss': '0.7419', 'grad_norm': '0.1729', 'learning_rate': '5.233e-05', 'epoch': '2.231'}                                                                                                                       
{'loss': '0.774', 'grad_norm': '0.1927', 'learning_rate': '5.188e-05', 'epoch': '2.237'}                                                                                                                        
{'loss': '0.7498', 'grad_norm': '0.1848', 'learning_rate': '5.144e-05', 'epoch': '2.244'}                                                                                                                       
{'loss': '0.7666', 'grad_norm': '0.1966', 'learning_rate': '5.1e-05', 'epoch': '2.251'}                                                                                                                         
{'loss': '0.7463', 'grad_norm': '0.1712', 'learning_rate': '5.055e-05', 'epoch': '2.257'}                                                                                                                       
{'loss': '0.7529', 'grad_norm': '0.2028', 'learning_rate': '5.011e-05', 'epoch': '2.264'}                                                                                                                       
{'loss': '0.7664', 'grad_norm': '0.2033', 'learning_rate': '4.967e-05', 'epoch': '2.27'}                                                                                                                        
{'loss': '0.7752', 'grad_norm': '0.1762', 'learning_rate': '4.922e-05', 'epoch': '2.277'}                                                                                                                       
{'loss': '0.7727', 'grad_norm': '0.2157', 'learning_rate': '4.878e-05', 'epoch': '2.284'}                                                                                                                       
{'loss': '0.7612', 'grad_norm': '0.1987', 'learning_rate': '4.834e-05', 'epoch': '2.29'}                                                                                                                        
{'loss': '0.7699', 'grad_norm': '0.1705', 'learning_rate': '4.789e-05', 'epoch': '2.297'}                                                                                                                       
{'loss': '0.757', 'grad_norm': '0.191', 'learning_rate': '4.745e-05', 'epoch': '2.303'}                                                                                                                         
{'loss': '0.7707', 'grad_norm': '0.1959', 'learning_rate': '4.701e-05', 'epoch': '2.31'}                                                                                                                        
{'loss': '0.7731', 'grad_norm': '0.1824', 'learning_rate': '4.656e-05', 'epoch': '2.316'}                                                                                                                       
{'loss': '0.7544', 'grad_norm': '0.2033', 'learning_rate': '4.612e-05', 'epoch': '2.323'}                                                                                                                       
{'loss': '0.7768', 'grad_norm': '0.1666', 'learning_rate': '4.568e-05', 'epoch': '2.33'}                                                                                                                        
{'loss': '0.7517', 'grad_norm': '0.2058', 'learning_rate': '4.523e-05', 'epoch': '2.336'}                                                                                                                       
{'loss': '0.7685', 'grad_norm': '0.1683', 'learning_rate': '4.479e-05', 'epoch': '2.343'}                                                                                                                       
{'loss': '0.7721', 'grad_norm': '0.1909', 'learning_rate': '4.435e-05', 'epoch': '2.349'}                                                                                                                       
{'loss': '0.7841', 'grad_norm': '0.1918', 'learning_rate': '4.39e-05', 'epoch': '2.356'}                                                                                                                        
{'loss': '0.7686', 'grad_norm': '0.2057', 'learning_rate': '4.346e-05', 'epoch': '2.363'}                                                                                                                       
{'loss': '0.753', 'grad_norm': '0.1668', 'learning_rate': '4.302e-05', 'epoch': '2.369'}                                                                                                                        
{'loss': '0.7507', 'grad_norm': '0.1711', 'learning_rate': '4.257e-05', 'epoch': '2.376'}                                                                                                                       
{'loss': '0.7886', 'grad_norm': '0.1704', 'learning_rate': '4.213e-05', 'epoch': '2.382'}                                                                                                                       
{'loss': '0.7558', 'grad_norm': '0.2103', 'learning_rate': '4.169e-05', 'epoch': '2.389'}                                                                                                                       
{'loss': '0.7484', 'grad_norm': '0.1946', 'learning_rate': '4.124e-05', 'epoch': '2.396'}                                                                                                                       
{'loss': '0.748', 'grad_norm': '0.1964', 'learning_rate': '4.08e-05', 'epoch': '2.402'}                                                                                                                         
{'loss': '0.7609', 'grad_norm': '0.1908', 'learning_rate': '4.035e-05', 'epoch': '2.409'}                                                                                                                       
{'loss': '0.7569', 'grad_norm': '0.2007', 'learning_rate': '3.991e-05', 'epoch': '2.415'}                                                                                                                       
{'loss': '0.7865', 'grad_norm': '0.1932', 'learning_rate': '3.947e-05', 'epoch': '2.422'}                                                                                                                       
{'loss': '0.7995', 'grad_norm': '0.1878', 'learning_rate': '3.902e-05', 'epoch': '2.429'}                                                                                                                       
{'loss': '0.7383', 'grad_norm': '0.1821', 'learning_rate': '3.858e-05', 'epoch': '2.435'}                                                                                                                       
{'loss': '0.7892', 'grad_norm': '0.1881', 'learning_rate': '3.814e-05', 'epoch': '2.442'}                                                                                                                       
{'loss': '0.766', 'grad_norm': '0.1653', 'learning_rate': '3.769e-05', 'epoch': '2.448'}                                                                                                                        
{'loss': '0.7584', 'grad_norm': '0.1849', 'learning_rate': '3.725e-05', 'epoch': '2.455'}                                                                                                                       
{'loss': '0.7626', 'grad_norm': '0.1962', 'learning_rate': '3.681e-05', 'epoch': '2.462'}                                                                                                                       
{'loss': '0.7711', 'grad_norm': '0.1874', 'learning_rate': '3.636e-05', 'epoch': '2.468'}                                                                                                                       
{'loss': '0.738', 'grad_norm': '0.2034', 'learning_rate': '3.592e-05', 'epoch': '2.475'}                                                                                                                        
{'loss': '0.7567', 'grad_norm': '0.1768', 'learning_rate': '3.548e-05', 'epoch': '2.481'}                                                                                                                       
{'loss': '0.7507', 'grad_norm': '0.1798', 'learning_rate': '3.503e-05', 'epoch': '2.488'}                                                                                                                       
{'loss': '0.747', 'grad_norm': '0.1844', 'learning_rate': '3.459e-05', 'epoch': '2.495'}                                                                                                                        
{'loss': '0.7788', 'grad_norm': '0.1789', 'learning_rate': '3.415e-05', 'epoch': '2.501'}                                                                                                                       
{'loss': '0.7421', 'grad_norm': '0.1835', 'learning_rate': '3.37e-05', 'epoch': '2.508'}                                                                                                                        
{'loss': '0.7512', 'grad_norm': '0.1811', 'learning_rate': '3.326e-05', 'epoch': '2.514'}                                                                                                                       
{'loss': '0.7753', 'grad_norm': '0.1866', 'learning_rate': '3.282e-05', 'epoch': '2.521'}                                                                                                                       
{'loss': '0.7824', 'grad_norm': '0.1773', 'learning_rate': '3.237e-05', 'epoch': '2.527'}                                                                                                                       
{'loss': '0.7612', 'grad_norm': '0.1717', 'learning_rate': '3.193e-05', 'epoch': '2.534'}                                                                                                                       
{'loss': '0.7498', 'grad_norm': '0.2083', 'learning_rate': '3.149e-05', 'epoch': '2.541'}                                                                                                                       
{'loss': '0.7637', 'grad_norm': '0.159', 'learning_rate': '3.104e-05', 'epoch': '2.547'}                                                                                                                        
{'loss': '0.7808', 'grad_norm': '0.1858', 'learning_rate': '3.06e-05', 'epoch': '2.554'}                                                                                                                        
{'loss': '0.7664', 'grad_norm': '0.1589', 'learning_rate': '3.016e-05', 'epoch': '2.56'}                                                                                                                        
{'loss': '0.7841', 'grad_norm': '0.1915', 'learning_rate': '2.971e-05', 'epoch': '2.567'}                                                                                                                       
{'loss': '0.7886', 'grad_norm': '0.2006', 'learning_rate': '2.927e-05', 'epoch': '2.574'}                                                                                                                       
{'loss': '0.7683', 'grad_norm': '0.1923', 'learning_rate': '2.882e-05', 'epoch': '2.58'}                                                                                                                        
{'loss': '0.7352', 'grad_norm': '0.1901', 'learning_rate': '2.838e-05', 'epoch': '2.587'}                                                                                                                       
{'loss': '0.7967', 'grad_norm': '0.1918', 'learning_rate': '2.794e-05', 'epoch': '2.593'}                                                                                                                       
{'loss': '0.7453', 'grad_norm': '0.1679', 'learning_rate': '2.749e-05', 'epoch': '2.6'}                                                                                                                         
{'loss': '0.793', 'grad_norm': '0.1772', 'learning_rate': '2.705e-05', 'epoch': '2.607'}                                                                                                                        
{'loss': '0.7721', 'grad_norm': '0.1951', 'learning_rate': '2.661e-05', 'epoch': '2.613'}                                                                                                                       
{'loss': '0.7964', 'grad_norm': '0.1852', 'learning_rate': '2.616e-05', 'epoch': '2.62'}                                                                                                                        
{'loss': '0.7305', 'grad_norm': '0.1805', 'learning_rate': '2.572e-05', 'epoch': '2.626'}                                                                                                                       
{'loss': '0.7712', 'grad_norm': '0.1971', 'learning_rate': '2.528e-05', 'epoch': '2.633'}                                                                                                                       
{'loss': '0.7219', 'grad_norm': '0.184', 'learning_rate': '2.483e-05', 'epoch': '2.64'}                                                                                                                         
{'loss': '0.7662', 'grad_norm': '0.1897', 'learning_rate': '2.439e-05', 'epoch': '2.646'}                                                                                                                       
{'loss': '0.7604', 'grad_norm': '0.1548', 'learning_rate': '2.395e-05', 'epoch': '2.653'}                                                                                                                       
{'loss': '0.7635', 'grad_norm': '0.176', 'learning_rate': '2.35e-05', 'epoch': '2.659'}                                                                                                                         
{'loss': '0.7407', 'grad_norm': '0.1731', 'learning_rate': '2.306e-05', 'epoch': '2.666'}                                                                                                                       
{'loss': '0.7402', 'grad_norm': '0.1891', 'learning_rate': '2.262e-05', 'epoch': '2.673'}                                                                                                                       
{'loss': '0.7409', 'grad_norm': '0.1796', 'learning_rate': '2.217e-05', 'epoch': '2.679'}                                                                                                                       
{'loss': '0.7464', 'grad_norm': '0.2044', 'learning_rate': '2.173e-05', 'epoch': '2.686'}                                                                                                                       
{'loss': '0.74', 'grad_norm': '0.2145', 'learning_rate': '2.129e-05', 'epoch': '2.692'}                                                                                                                         
{'loss': '0.7704', 'grad_norm': '0.2017', 'learning_rate': '2.084e-05', 'epoch': '2.699'}                                                                                                                       
{'loss': '0.7802', 'grad_norm': '0.2083', 'learning_rate': '2.04e-05', 'epoch': '2.705'}                                                                                                                        
{'loss': '0.774', 'grad_norm': '0.185', 'learning_rate': '1.996e-05', 'epoch': '2.712'}                                                                                                                         
{'loss': '0.7406', 'grad_norm': '0.199', 'learning_rate': '1.951e-05', 'epoch': '2.719'}                                                                                                                        
{'loss': '0.734', 'grad_norm': '0.1846', 'learning_rate': '1.907e-05', 'epoch': '2.725'}                                                                                                                        
{'loss': '0.7505', 'grad_norm': '0.1823', 'learning_rate': '1.863e-05', 'epoch': '2.732'}                                                                                                                       
{'loss': '0.7969', 'grad_norm': '0.1764', 'learning_rate': '1.818e-05', 'epoch': '2.738'}                                                                                                                       
{'loss': '0.752', 'grad_norm': '0.2157', 'learning_rate': '1.774e-05', 'epoch': '2.745'}                                                                                                                        
{'loss': '0.7609', 'grad_norm': '0.1691', 'learning_rate': '1.729e-05', 'epoch': '2.752'}                                                                                                                       
{'loss': '0.7353', 'grad_norm': '0.1887', 'learning_rate': '1.685e-05', 'epoch': '2.758'}                                                                                                                       
{'loss': '0.7177', 'grad_norm': '0.1881', 'learning_rate': '1.641e-05', 'epoch': '2.765'}                                                                                                                       
{'loss': '0.7724', 'grad_norm': '0.1715', 'learning_rate': '1.596e-05', 'epoch': '2.771'}                                                                                                                       
{'loss': '0.7413', 'grad_norm': '0.2041', 'learning_rate': '1.552e-05', 'epoch': '2.778'}                                                                                                                       
{'loss': '0.733', 'grad_norm': '0.1882', 'learning_rate': '1.508e-05', 'epoch': '2.785'}                                                                                                                        
{'loss': '0.7176', 'grad_norm': '0.1982', 'learning_rate': '1.463e-05', 'epoch': '2.791'}                                                                                                                       
{'loss': '0.7493', 'grad_norm': '0.1749', 'learning_rate': '1.419e-05', 'epoch': '2.798'}                                                                                                                       
{'loss': '0.7475', 'grad_norm': '0.1783', 'learning_rate': '1.375e-05', 'epoch': '2.804'}                                                                                                                       
{'loss': '0.7771', 'grad_norm': '0.1818', 'learning_rate': '1.33e-05', 'epoch': '2.811'}                                                                                                                        
{'loss': '0.7569', 'grad_norm': '0.1767', 'learning_rate': '1.286e-05', 'epoch': '2.818'}                                                                                                                       
{'loss': '0.7371', 'grad_norm': '0.2001', 'learning_rate': '1.242e-05', 'epoch': '2.824'}                                                                                                                       
{'loss': '0.7896', 'grad_norm': '0.217', 'learning_rate': '1.197e-05', 'epoch': '2.831'}                                                                                                                        
{'loss': '0.7597', 'grad_norm': '0.1917', 'learning_rate': '1.153e-05', 'epoch': '2.837'}                                                                                                                       
{'loss': '0.7682', 'grad_norm': '0.181', 'learning_rate': '1.109e-05', 'epoch': '2.844'}                                                                                                                        
{'loss': '0.7628', 'grad_norm': '0.1919', 'learning_rate': '1.064e-05', 'epoch': '2.851'}                                                                                                                       
{'loss': '0.7834', 'grad_norm': '0.1821', 'learning_rate': '1.02e-05', 'epoch': '2.857'}                                                                                                                        
{'loss': '0.8219', 'grad_norm': '0.1887', 'learning_rate': '9.756e-06', 'epoch': '2.864'}                                                                                                                       
{'loss': '0.7805', 'grad_norm': '0.1778', 'learning_rate': '9.313e-06', 'epoch': '2.87'}                                                                                                                        
{'loss': '0.7769', 'grad_norm': '0.1853', 'learning_rate': '8.869e-06', 'epoch': '2.877'}                                                                                                                       
{'loss': '0.7762', 'grad_norm': '0.1708', 'learning_rate': '8.426e-06', 'epoch': '2.884'}                                                                                                                       
{'loss': '0.7948', 'grad_norm': '0.2151', 'learning_rate': '7.982e-06', 'epoch': '2.89'}                                                                                                                        
{'loss': '0.7377', 'grad_norm': '0.1839', 'learning_rate': '7.539e-06', 'epoch': '2.897'}                                                                                                                       
{'loss': '0.7545', 'grad_norm': '0.1785', 'learning_rate': '7.095e-06', 'epoch': '2.903'}                                                                                                                       
{'loss': '0.7502', 'grad_norm': '0.206', 'learning_rate': '6.652e-06', 'epoch': '2.91'}                                                                                                                         
{'loss': '0.7844', 'grad_norm': '0.1994', 'learning_rate': '6.208e-06', 'epoch': '2.916'}                                                                                                                       
{'loss': '0.7218', 'grad_norm': '0.2042', 'learning_rate': '5.765e-06', 'epoch': '2.923'}                                                                                                                       
{'loss': '0.7731', 'grad_norm': '0.1744', 'learning_rate': '5.322e-06', 'epoch': '2.93'}                                                                                                                        
{'loss': '0.7631', 'grad_norm': '0.1952', 'learning_rate': '4.878e-06', 'epoch': '2.936'}                                                                                                                       
{'loss': '0.7729', 'grad_norm': '0.187', 'learning_rate': '4.435e-06', 'epoch': '2.943'}                                                                                                                        
{'loss': '0.7282', 'grad_norm': '0.1637', 'learning_rate': '3.991e-06', 'epoch': '2.949'}                                                                                                                       
{'loss': '0.7806', 'grad_norm': '0.1653', 'learning_rate': '3.548e-06', 'epoch': '2.956'}                                                                                                                       
{'loss': '0.7228', 'grad_norm': '0.1962', 'learning_rate': '3.104e-06', 'epoch': '2.963'}                                                                                                                       
{'loss': '0.7629', 'grad_norm': '0.1911', 'learning_rate': '2.661e-06', 'epoch': '2.969'}                                                                                                                       
{'loss': '0.7626', 'grad_norm': '0.1829', 'learning_rate': '2.217e-06', 'epoch': '2.976'}                                                                                                                       
{'loss': '0.7411', 'grad_norm': '0.1794', 'learning_rate': '1.774e-06', 'epoch': '2.982'}                                                                                                                       
{'loss': '0.7706', 'grad_norm': '0.1891', 'learning_rate': '1.33e-06', 'epoch': '2.989'}                                                                                                                        
{'loss': '0.7511', 'grad_norm': '0.1843', 'learning_rate': '8.869e-07', 'epoch': '2.996'}                                                                                                                       
{'loss': '0.7497', 'grad_norm': '0.2436', 'learning_rate': '4.435e-07', 'epoch': '3'}                                                                                                                           
{'train_runtime': '1840', 'train_samples_per_second': '16.31', 'train_steps_per_second': '0.248', 'train_loss': '0.8071', 'epoch': '3'}                                                                         
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456/456 [30:39<00:00,  4.03s/it]
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 527/527 [00:00<00:00, 3.13MB/s]
Processing Files (0 / 0)      : |                                                                                                                                                  |  0.00B /  0.00B           ^CCancellation requested; stopping current tasks.                                                                                                                                   |  0.00B /  0.00B            
Processing Files (0 / 0)      :   0%|                                                                                                                                              |  0.00B / 12.8MB,  0.00B/s  
New Data Upload               : |                                                                                                                                                  |  0.00B /  0.00B,  0.00B/s  
Processing Files (1 / 1)      : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.8MB / 12.8MB, 3.77MB/s  
New Data Upload               : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.8MB / 12.8MB, 3.77MB/s  
  ...adapter_model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.8MB / 12.8MB            
Saved model to https://huggingface.co/DesKate/qwen_3.5_0.8b-numbers_control
Processing Files (1 / 1)      : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20.0MB / 20.0MB,  0.00B/s  
New Data Upload               : |                                                                                                                                                  |  0.00B /  0.00B,  0.00B/s  
  ...mpnwy11c_n/tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20.0MB / 20.0MB            
Processing Files (1 / 1)      : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.8MB / 12.8MB,  0.00B/s  
New Data Upload               : |                                                                                                                                                  |  0.00B /  0.00B,  0.00B/s  
  ...adapter_model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12.8MB / 12.8MB            
No files have been modified since last commit. Skipping to prevent empty commit.
[huggingface_hub.hf_api|WARNING]No files have been modified since last commit. Skipping to prevent empty commit.
Saved model to https://huggingface.co/DesKate/qwen_3.5_0.8b-numbers_control
Processing Files (1 / 1)      : 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20.0MB / 20.0MB,  0.00B/s  
New Data Upload               : |                                                                                                                                                  |  0.00B /  0.00B,  0.00B/s  
  ...mpyoxay150/tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20.0MB / 20.0MB            
No files have been modified since last commit. Skipping to prevent empty commit.
[huggingface_hub.hf_api|WARNING]No files have been modified since last commit. Skipping to prevent empty commit.
2026-04-26 18:19:09.804 | INFO     | sl.finetuning.services:_run_unsloth_finetuning_job:119 - ✅ Model uploaded to HuggingFace Hub: DesKate/qwen_3.5_0.8b-numbers_control
2026-04-26 18:19:09.811 | SUCCESS  | sl.finetuning.services:run_finetuning_job:232 - Finetuning job completed successfully! External ID: DesKate/qwen_3.5_0.8b-numbers_control
2026-04-26 18:19:09.811 | INFO     | __main__:main:84 - Saved output to data/demo/model_qwen3.5-0.8b_ft_control.json
2026-04-26 18:19:09.811 | SUCCESS  | __main__:main:85 - Fine-tuning job completed successfully!
Traceback (most recent call last):
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/base_events.py", line 687, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/turing/subliminal-learning/scripts/run_finetuning_job.py", line 94, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 123, in run
    raise KeyboardInterrupt()
KeyboardInterrupt
(.venv) turing@turing-Predator-PH18-73:~/subliminal-learning$ 