import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer, SFTConfig
import deepspeed
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loading a YAML Configuration File"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Creating a DeepSpeed ​​Configuration"""
    return {
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": True,
  "bf16": {
    "enabled": "auto"
  },
#   "zero_optimization": {
#         "stage": 1
#     },
  "zero_optimization": {
            "stage": 2,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
#   "zero_optimization": {
#     "stage": 3,
#     "overlap_comm": True,
#     "contiguous_gradients": True,
#     "sub_group_size": 1e9,
#     "reduce_bucket_size": "auto",
#     "stage3_prefetch_bucket_size": "auto",
#     "stage3_param_persistence_threshold": "auto",
#     "stage3_max_live_parameters": 1e9,
#     "stage3_max_reuse_distance": 1e9,
#     "stage3_gather_16bit_weights_on_model_save": True
#   },
}

def prepare_model(config: Dict[str, Any]):
    """Prepare the model and tokenizer according to the configuration"""

    # Setting torch dtype
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    
    # Loading the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch_dtype,
        trust_remote_code=config['model']['trust_remote_code'],
        use_cache=False,
        attn_implementation="flash_attention_2"  # Flash Attention 2.0
    )

    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configuring LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )

    # Preparing the model for training
    model = get_peft_model(model, lora_config)
    
    # Print the number of trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():

    from transformers import DataCollatorWithPadding

    # 1. Loading configuration, model and tokenizer
    config = load_config('config/config_lora_r1_7b.yaml')

    # 2. Setting training parameters
    training_args = SFTConfig(
        **config['training'],
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        max_seq_length=5000,
        label_names=["labels"]
    )

    model, tokenizer = prepare_model(config)

    # 3. Load the original dataset
    dataset = load_dataset("Zigeng/CoT-Verification-340k",split="train")


    # 4. Format each sample, generate the complete text and record the number of tokens in the prompt section
    def format_example(example):
        texts = []
        prompt_lengths = []
        for question, response in zip(example["prompt"], example["response"]):
            messages = [{"role": "user", "content": question}]
            #  prompt text
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # response test
            answer_text = response + "\n</think>" + tokenizer.eos_token
            # complete text
            full_text = prompt_text + answer_text
            texts.append(full_text)
            # Calculate the number of tokens in the prompt part 
            prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(prompt_token_ids))
        return {"text": texts, "prompt_length": prompt_lengths}

    dataset = dataset.map(format_example, batched=True)

    # 5. Tokenize the complete text and modify labels
    def tokenize_and_mask(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=training_args.max_seq_length,
            add_special_tokens=False  
        )
        labels = []
        for ids, pl in zip(tokenized["input_ids"], example["prompt_length"]):
            lab = ids.copy()
            # Set the label of the prompt part to -100 (do not calculate loss)
            lab[:pl] = [-100] * pl
            labels.append(lab)
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(tokenize_and_mask, batched=True)

    # 6. Create a custom data collator to ensure dynamic padding to the maximum length in the batch
    class DataCollatorForChat(DataCollatorWithPadding):
        def __call__(self, features):
            labels = [feature.pop("labels") for feature in features]
            batch = super().__call__(features)
            max_label_len = max(len(label) for label in labels)
            padded_labels = [
                [-100] * (max_label_len - len(label))+label
                for label in labels
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            return batch

    # data collator
    data_collator = DataCollatorForChat(
        tokenizer=tokenizer,
        padding=True, 
        return_tensors="pt"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,  
    )

    trainer.train()
    
    model = trainer.model

if __name__ == "__main__":
    main()