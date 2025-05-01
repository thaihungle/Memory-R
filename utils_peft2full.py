import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Load a PEFT adapter checkpoint and save the full model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to the PEFT adapter checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the full model")
    
    args = parser.parse_args()

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load adapter and merge with base model
    model = PeftModel.from_pretrained(base_model, args.adapter_checkpoint)
    model = model.merge_and_unload()  # Merge LoRA layers into base model

    # Save the full model
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"Full model saved at: {args.output_path}")

if __name__ == "__main__":
    main()
