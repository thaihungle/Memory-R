# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import transformers
import tqdm

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from utils_gsm8k import *
import argparse
import torch
import random
import numpy as np
import torch.distributed as dist
import wandb

# use os and sys to get the path to the repo
import os
import sys
import json
PATH_TO_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class SaveTrainingStatsCallback(TrainerCallback):
    def __init__(self, output_file="training_stats.jsonl"):
        self.output_file = output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.output_file, "a") as f:
                json.dump(logs, f)
                f.write("\n")

class WandbTrainingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

# model_size = "0.5B"
# model_name = f"meta-llama/Llama-3.2-{model_size}-Instruct"
# model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"

# Reward functions
# def novelty_reward_func_explore(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     # extracted_responses = [extract_xml_reasoning(r) for r in responses]
#     extracted_responses = responses
#     rewards = []
#     for r in extracted_responses:
#         s = knn_memory_explore.novelty_score_mean(q, r, k=args.k)
#         rewards.append(s*0.05)
#         knn_memory_explore.insert_pair(q, r)
#     return rewards

def generate_gsm8k(
    model,
    tokenizer,
    tokenized_samples,
    batch_size,
    max_completion_length
):
    # run eval on main
    if dist.get_rank() == 0:
      device = model.device
      predictions = []
      generation_config = transformers.GenerationConfig(
          max_new_tokens=max_completion_length,
          do_sample=False,
          repetition_penalty=1.0,
          eos_token_id=tokenizer.eos_token_id,
          pad_token_id=tokenizer.pad_token_id,
      )
      model.eval()
      count = len(tokenized_samples)
      
      status = tqdm.tqdm(tokenized_samples, desc=f"Correct: 0/{count}")
      for i in range(0, count, batch_size):
        batches = tokenized_samples[i:i+batch_size]
        with torch.inference_mode():
            longest = max(len(b[0]) for b in batches)

            # pad to longest on left side for decoder
            padded_input_ids = torch.stack([
                torch.tensor([tokenizer.pad_token_id] * (longest - len(ids)) + ids)
                for ids, _ in batches
            ]).to(device)
            # ignore pad token when generating
            attn_mask = torch.stack([
                tokens.ne(tokenizer.pad_token_id) for tokens in padded_input_ids
            ]).to(device)

            output = model.generate(
                input_ids=padded_input_ids,
                attention_mask=attn_mask,
                generation_config=generation_config,
            )

            for i, generated in enumerate(output):
              response = tokenizer.decode(
                  generated[len(padded_input_ids[i]) :], skip_special_tokens=True
              )

              prediction = extract_xml_answer(response)
              predictions.append(batches[i][1] == prediction)

            status.update(batch_size)
            status.set_description(f"Correct: {sum(predictions)}/{count}")

      return np.mean(predictions)

    return 0

def tokenize_validation(tokenizer, samples, max_prompt_length):
    tokenized_samples = []
    for sample in samples:
        prompt = sample["prompt"]
        answer = sample['answer']
        ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            truncation=False,
            max_length=max_prompt_length,
        )
        tokenized_samples.append((ids, answer))
    return tokenized_samples

class EvalTrainer(GRPOTrainer):
    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ):
        tokenized_samples = tokenize_validation(self.processing_class, self.eval_dataset, self.args.max_prompt_length)
        eval_acc = generate_gsm8k(self.model, self.processing_class, tokenized_samples, self.args.per_device_eval_batch_size, self.args.max_completion_length)

        output = {
            f"{metric_key_prefix}_accuracy": eval_acc,
            "epoch": self.state.epoch,
        }

        self.log(output)
        wandb.log(output)  # Log evaluation results to wandb

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output
        )

        return output

def novelty_reward_func_explore(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_reasoning = [extract_xml_reasoning(r) for r in responses]
    # extracted_responses = responses
    rewards = []
    # contents = [completion[0]["content"] for completion in completions]
    # for c in contents:
    #     if count_xml(c)>0:
    #         knn_memory_exploit.insert_pair(q, c)
    for r, a, c in zip(extracted_responses, answer, extracted_reasoning):
        if r!=a:
            knn_memory_explore.insert_pair(q, c)
            # print("insert ..")
        s = knn_memory_explore.novelty_score_max(q, c, k=args.k)
        if s is not None:
            rewards.append(s)
        else:
            rewards.append(0)
    return rewards

def novelty_reward_func_exploit(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_reasoning = [extract_xml_reasoning(r) for r in responses]
    # extracted_responses = responses
    rewards = []
    # contents = [completion[0]["content"] for completion in completions]
    # for c in contents:
    #     if count_xml(c)>0:
    #         knn_memory_exploit.insert_pair(q, c)
    for r, a, c in zip(extracted_responses, answer, extracted_reasoning):
        if r==a:
            knn_memory_exploit.insert_pair(q, c)
            # print("insert ..")
        s = knn_memory_exploit.novelty_score_mean(q, c, k=args.k)
        if s is not None:
            rewards.append(1-s)
        else:
            rewards.append(0)
    return rewards

def novelty_reward_func_exploit_bypass_template_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    rewards = []
    extracted_responses = []
    for r in responses:
        all_numbers = re.findall('\d+[.,]?\d*\s', r)
        if len(all_numbers) > 0:
            extracted_responses.append(all_numbers[-1].replace('.', '').replace(',', '').replace('\n', ''))
        else:
            extracted_responses.append(f"-1uiekc7") # no reward
            # print(f"all_numbers = {all_numbers}, check this response : {r}")
    for r, a in zip(extracted_responses, answer):
        s1 = knn_memory_exploit.novelty_score_mean(q, r, k=args.k)
        s2 = knn_memory_explore.novelty_score_mean(q, r, k=args.k)
        if r==a:
            knn_memory_exploit.insert_pair(q, r)
        else:
            knn_memory_explore.insert_pair(q, r)
        
        rewards.append((1-s1)*1)

    return rewards

def entropy_reward_func(prompts, completions, answer, **kwargs)-> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_reasoning(r) for r in responses]
    # extracted_responses = responses
    rewards = []
    for r in extracted_responses:
        s  =  entropy.update_and_score(r)
        rewards.append(s)
    return rewards

def lexical_reward_func(prompts, completions, answer, **kwargs)-> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_reasoning(r) for r in responses]
    # extracted_responses = responses
    rewards = []
    for r in extracted_responses:
        ltree.insert(r)
        s  =  ltree.compute_novelty(r)
        rewards.append(s*0.1)
    return rewards


if __name__ == '__main__':
   

    parser = argparse.ArgumentParser(description='Train GRPO')
    parser.add_argument('--model_name', type=str, required=True, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--use_ir', type=str, required=True, default="knn")
    parser.add_argument('--k', type=int, required=True, default=1)
    parser.add_argument('--num_shots', type=int, required=True, default=0)
    parser.add_argument('--nepochs', type=int, required=True, default=1)
    parser.add_argument('--seed', type=int, required=True, default=2025)
    parser.add_argument('--bs', type=int, required=False, default=2)
    parser.add_argument('--gc', type=int, required=False, default=8)
    parser.add_argument('--L', type=int, required=False, default=200)
    parser.add_argument('--do_eval', type=int, required=False, default=0)

    args = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    wandb.init(
        project="GRPO-training",
        name=f"{args.model_name}ir{args.use_ir}-k{args.k}-shots{args.num_shots}-seed{args.seed}",
        config=vars(args),
    )

    reward_list = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func]

    # if "Llama" in args.model_name:
    #     reward_list[-1] =correctness_reward_bypass_template_func
        
    
    if "knn+" in args.use_ir:
        from rewards.ir_knn_st import FastKNNMemory
        knn_memory_exploit = FastKNNMemory(max_keys=10000, max_values=100, history_size=100, anneal_rate=1)
        knn_memory_explore = FastKNNMemory(max_keys=10000, max_values=100, history_size=100, anneal_rate=1, explore_phase=500)
        # if "Llama" in args.model_name:
        #     reward_list.append(novelty_reward_func_exploit_bypass_template_func)
        # else:
        reward_list.append(novelty_reward_func_exploit)
        reward_list.append(novelty_reward_func_explore)
    elif "knn" in args.use_ir:
        from rewards.ir_knn_st import FastKNNMemory
        knn_memory_exploit = FastKNNMemory(max_keys=10000, max_values=1000, history_size=100, anneal_rate=1)
        # if "Llama" in args.model_name:
        #     reward_list.append(novelty_reward_func_exploit_bypass_template_func)
        # else:
        reward_list.append(novelty_reward_func_exploit)
    if "entropy" in args.use_ir:
        from rewards.ir_entropy import EntropyNoveltyEstimator
        entropy = EntropyNoveltyEstimator(history_size=1000)
        reward_list.append(entropy_reward_func)
    if "ltree" in args.use_ir:
        from rewards.ir_lexical_novelty import TreeNoveltyEstimator
        ltree = TreeNoveltyEstimator(history_size=50)
        reward_list.append(lexical_reward_func)
    if 'cosine' in args.use_ir:
        reward_list.append(get_cosine_scaled_reward(max_len=args.L))


    train_dataset = get_gsm8k_questions(split = "train", num_shots=args.num_shots).shuffle(seed=args.seed) 
    eval_dataset = get_gsm8k_questions(split = "test")


    output_dir = f"{PATH_TO_REPO}/output/{args.model_name}-GRPO-{args.use_ir}-{args.k}-{args.num_shots}-seed{args.seed}"
    run_name = f"{args.model_name}-GRPO-gsm8k-{args.use_ir}-{args.k}-{args.num_shots}-seed{args.seed}"

    # assert the output directory exists, otherwise create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    do_eval = True
    eval_strategy ="steps"
    if args.do_eval==0:
        do_eval = False
        eval_strategy = "no"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy=eval_strategy,
        eval_steps=50,
        do_eval=do_eval,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gc,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=args.L,
        num_train_epochs=args.nepochs,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=.3,
        vllm_device="cuda:0",
        report_to="tensorboard" 
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        use_cache=False,
    ).to("cuda")
            
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # use peft at your own risk; not working for me with multi-GPU training
    trainer = EvalTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_list,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    save_training_path = f"{PATH_TO_REPO}/output/{args.model_name}-{args.use_ir}-{args.k}-{args.num_shots}-seed{args.seed}/"
    save_training_filepath = f"{save_training_path}/training_stats_seed{args.seed}.jsonl"
    if not os.path.exists(save_training_path):
        os.makedirs(save_training_path)

    trainer.add_callback(WandbTrainingCallback()) 
    trainer.add_callback(SaveTrainingStatsCallback(output_file=f"{PATH_TO_REPO}/output/{args.model_name}-GRPO-{args.use_ir}-{args.k}-{args.num_shots}-seed{args.seed}/training_stats_seed{args.seed}.jsonl"))

    trainer.train()
