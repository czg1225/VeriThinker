
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    PeftModel,
)
from typing import Dict, Any


def main():

    name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Loading the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map='auto',
        torch_dtype='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        name,
    )
    # Loading LoRA weights
    peft_model = PeftModel.from_pretrained(base_model, "model/R1_VeriThinker_7B_lora")

    # lora merging
    merged_model = peft_model.merge_and_unload()

    # Save the merged complete model
    merged_model.save_pretrained("model/R1_VeriThinker_7B")
    tokenizer.save_pretrained("model/R1_VeriThinker_7B")


if __name__ == "__main__":
    main()