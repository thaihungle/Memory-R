# Memory-R

## Reference:
- GSM8K: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
- Other Math: https://github.com/huggingface/open-r1/tree/main/src/open_r1

## Main idea
- Message: create self-motivated LLMs that can explore without the need of ground-truth rewards
- Use knn memory to create novelty-based intrinsic reward. Motivation from RL literature: https://deepmind.google/discover/blog/agent57-outperforming-the-human-atari-benchmark/
- Encourage better RL exploration --> sample efficiency, no need to train for long and with lots of data
- Goal: help boost small LLM performance where sample efficiency and low-resource are key

## Setup
```bash
# Install Python
conda create -n llmrl python=3.11
conda activate llmrl
# Install other dependencies
pip install -r requirements.txt
```


## Experiments (on going)
- Only focus on small LLMs that require 1 GPU for training with RL
- With full training: can go upto 1.5-1.7B
- With LORA: can go upto 3B



### Dataset: GSM8K
#### Training
- Qwen2.5-0.5B-Instruct: 
```bash
python run_gsm8k.py --model_name=Qwen/Qwen2.5-0.5B-Instruct --use_ir=no --num_shots=0 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200 # pure RL
python run_gsm8k.py --model_name=Qwen/Qwen2.5-0.5B-Instruct --use_ir=knn --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200 # Memory (Only Exploit)
python run_gsm8k.py --model_name=Qwen/Qwen2.5-0.5B-Instruct --use_ir=knn+ --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200 # Memory (Exploit+Explore)
```
- Llama3.2-1B-Instruct (note: needs num_shots=1 to have at least 1 correct answer for learning):
```bash
python run_gsm8k.py --model_name=meta-llama/Llama-3.2-1B-Instruct --use_ir=no --num_shots=1 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200
python run_gsm8k.py --model_name=meta-llama/Llama-3.2-1B-Instruct --use_ir=knn --k=1 --num_shots=1 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200
python run_gsm8k.py --model_name=meta-llama/Llama-3.2-1B-Instruct --use_ir=knn+ --k=1 --num_shots=1 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200
```
From now on, to fit 1 GPU, batch size is set to 1.
- Falcon3-1B-Instruct: 
```bash
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=no --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=knn --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=knn+ --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
```
- SmolLM-1.7B-Instruct: 
```bash
python run_gsm8k.py --model_name=HuggingFaceTB/SmolLM2-1.7B-Instruct --use_ir=no --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=HuggingFaceTB/SmolLM2-1.7B-Instruct --use_ir=knn --k=1 --num_shots=0 --nepochs=2 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=HuggingFaceTB/SmolLM2-1.7B-Instruct --use_ir=knn+ --k=1 --num_shots=0 --nepochs=2 --seed 0 --bs 1 --gc 16 --L 200
```
- Qwen2.5-1.5B-Instruct: 
```bash
python run_gsm8k.py --model_name=Qwen/Qwen2.5-1.5B-Instruct --use_ir=no --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 300
python run_gsm8k.py --model_name=Qwen/Qwen2.5-1.5B-Instruct --use_ir=knn --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 300
python run_gsm8k.py --model_name=Qwen/Qwen2.5-1.5B-Instruct --use_ir=knn+ --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 300
```

#### Evaluation
- Strict using LightEval: 
```bash
python run_eval.py --task=gsm8k --model_name=path/to/model/
```
- Medium using Eval loop of the Trainer (the results tend to be higher than LightEval)
- Loose using QwenEval (the results will be higher, match SOTA, but performance gain tends to be less): 
```bash
cd evaluation
bash sh/eval_gsm8k.sh qwen25-math-cot path/to/model/
```


<!-- 
## <a name="todo"></a> 🤝 Things to Do
- [X] Train R1-style on GSM8K. 2 options: full training (python run_gsm8k_full.py) or PEFT (python run_gsm8k_peft.py). Update: Now focus on full training in run_gsm8k
- [X] Train R1-style on countdown task, supporting multiple GPUs: full (python run_countdown.py). So far memory does not work with this dataset
- [X] Evalutation on GSM8K, need full weights (python run_eval.py --task gsm8k). However, sometimes got lighteval parsing error (seems like using non-instruct LLM base model makes the error)
- [X] Implement simple KNN Memory full response level (ir_knn_st.py)
- [X] Implement simple KNN Memory sentence level (ir_knn_st2.py), this one seems to train better but generalize worse. 
- [X] Train R1-style on GSM8K with KNN novelty intrinsic rewards: full training (python run_gsm8k.py). Still testing, not sure it will help. Insights on gsm8k: helps significantly with some LLMs (7-8%), some LLMs small gain (1%) 
- [X] Dung: find the evaluation that can reproduce SOTA (Qwen eval)
- [X] Other intrinsic rewards: surprise, entropy ...: Memory still the best
- [X] Add eval loop to training
- [X] Fix lighteval errors, support peft evaluation ...
- [X] Implement Cosine Reward
- [X] Run on other math  AI MO
- [ ] **Confirm results on GSM8K (light-eval, qwen-eval across checkpoints)**
- [ ] **Run Baseline SFT GSM8K**
- [ ] **Run Baseline Cosine GSM8K**
- [ ] Fix Eval loop accuracy seems wrong when num_shots>0
- [ ] **Run on MMLU**
- [ ] **Run on LiveCodeBench**
 -->
