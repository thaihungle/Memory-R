# Memory-R

## Code Reference:
- GSM8K: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

## Setup
```bash
# Install Python
conda create -n memoryr python=3.11
conda activate memoryr
# Install other dependencies
pip install -r requirements.txt
```


## Experiments

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
- Falcon3-1B-Instruct: 
```bash
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=no --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=knn --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
python run_gsm8k.py --model_name=tiiuae/Falcon3-1B-Instruct --use_ir=knn+ --k=1 --num_shots=0 --nepochs=1 --seed 0 --bs 1 --gc 16 --L 200
```
#### Evaluation
- Strict using LightEval: 
```bash
python run_eval.py --task=gsm8k --model_name=path/to/model/
```
