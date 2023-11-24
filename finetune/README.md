# Finetune Pipeline

## Parameters
```bash

Arguments for finetune.py:
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library).
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the datasets library).
  --train_file TRAIN_FILE
                        A csv or a json file containing the training data.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models.
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name
  --do_eval             Run evaluation on the dev set.
  --eval_file EVAL_FILE
                        A csv or a json file containing the evaluation data.
  --eval_dataset_name EVAL_DATASET_NAME
                        The name of the dataset to use (via the datasets library).
  --eval_epochs EVAL_EPOCHS
                        Number of epochs between evaluations.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for evaluation.
  --use_lora            If passed, will use LORA (low-rank parameter-efficient training) to train the model.
  --lora_rank LORA_RANK
                        The rank of lora.
  --lora_alpha LORA_ALPHA
                        The alpha parameter of lora.
  --lora_dropout LORA_DROPOUT
                        The dropout rate of lora modules.
  --use_flash_attn      If passed, will use flash attention to train the model.
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --use_slow_tokenizer  If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total sequence length (prompt+completion) of each training example.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup period) to use.
  --weight_decay WEIGHT_DECAY
                        Weight decay to use.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If provided, overrides num_train_epochs.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.
  --warmup_ratio WARMUP_RATIO
                        Ratio of total training steps used for warmup.
  --output_dir OUTPUT_DIR
                        Where to store the final model.
  --seed SEED           A seed for reproducible training.
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --checkpointing_steps CHECKPOINTING_STEPS
                        Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.
  --logging_steps LOGGING_STEPS
                        Log the training loss and learning rate every logging_steps steps.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        If the training should continue from a checkpoint folder.
  --with_tracking       Whether to enable experiment trackers for logging.
  --report_to REPORT_TO
                        The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`,
                        `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.Only applicable when
                        `--with_tracking` is passed.
  --low_cpu_mem_usage   It is an option to create the model as an empty shell, then only materialize its parameters when the
                        pretrained weights are loaded.If passed, LLM loading time and RAM consumption will be benefited.
  --gradient_checkpointing
                        Turn on gradient checkpointing. Saves memory but slows training.
  --use_qlora           Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed.
  --clip_grad_norm CLIP_GRAD_NORM
                        Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).
  --use_8bit_optimizer  Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).
```