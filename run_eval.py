import os
import sys
import argparse
import torch
import random
import numpy as np
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']="1"
# model_size = "3B"
# model_name = f"meta-llama/Llama-3.2-{model_size}-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "outputs/Qwen-1.5B-GRPO/checkpoint-300"
# model_name = "outputs/Qwen-1.5B-GRPO-Mem/checkpoint-500"
# model_name = "Qwen/Qwen2.5-Math-1.5B"
# model_name = "Qwen/Qwen2.5-0.5B"



if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    parser = argparse.ArgumentParser(description='LightEval')
    parser.add_argument('--model_name', type=str, required=True, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument('--num_shots', type=int, required=True, default=0)
    parser.add_argument('--task_name', type=str, required=True, default="gsm8k")
    parser.add_argument('--model_seed', type=int, required=True, default=0, choices=[0, 1, 2], help="This is the seed for the model weights")
    args = parser.parse_args()
    SYSTEM_PROMPT = """
    Respond in the following format:

    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    MODEL_ARGS=f"pretrained={args.model_name},dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.8"
    OUTPUT_DIR=f"./dai_results/{args.model_name}"

    os.system(f'lighteval vllm {MODEL_ARGS} "custom|{args.task_name}|{args.num_shots}|1" '
            f'--custom-tasks ./evaluate.py '
            f'--use-chat-template '
            f'--system-prompt="{SYSTEM_PROMPT}" '
            f'--output-dir="{OUTPUT_DIR}" || true')