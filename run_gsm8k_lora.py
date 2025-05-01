import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils_gsm8k import *
from ir_knn_st import FastKNNMemory

import argparse
import torch
import random
import numpy as np



# model_size = "0.5B"
# model_name = f"meta-llama/Llama-3.2-{model_size}-Instruct"
# model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"

# Reward functions
def novelty_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_reasoning(r) for r in responses]
    rewards = []
    for r in extracted_responses:
        s = knn_memory.novelty_score_mean(q, r, k=10)
        rewards.append(s)
    for r in extracted_responses:
        knn_memory.insert_pair(q, r)
    return rewards

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Train GRPO')
    parser.add_argument('--model_name', type=str, required=True, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--use_mem', type=int, required=True, default=0)

    args = parser.parse_args()
    reward_list = [
            # xmlcount_reward_func,
            # soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func]
    if args.use_mem:
        from knn_st import FastKNNMemory
        knn_memory = FastKNNMemory(max_keys=10000, max_values=100, history_size=1000, anneal_rate=1)
        reward_list.append(novelty_reward_func)

    train_dataset = get_gsm8k_questions(split = "train").shuffle(seed=42)
    eval_dataset = get_gsm8k_questions(split = "test")



    output_dir = f"outputs/Qwen-{args.model_name}-GRPO-lora-{args.use_mem}"
    run_name = f"{args.model_name}-GRPO-gsm8k-lora-{args.use_mem}"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=2e-5,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=6,
        max_prompt_length=256,
        max_completion_length=300,
        num_train_epochs=2,
        save_steps=100,
        max_grad_norm=0.1,
        report_to='tensorboard',
        log_on_each_node=False,
        # use_vllm=True,
        # vllm_device='auto',
    )


    rank = 16
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=rank*2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        bias='none',
        lora_dropout=0.05,
    )


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_list,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

model.base_model.save_pretrained(f"./outputs/{args.model_name}")
