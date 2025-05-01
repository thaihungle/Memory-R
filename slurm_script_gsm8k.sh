#!/bin/bash


#SBATCH --output ./sbatch/job%j.out # Output file name
#SBATCH --error ./sbatch/job%j.err # Error log file name
## Below is for requesting the resource you want
#SBATCH --nodes=1 # Number of nodes required
## SBATCH --exclusive  --gres=gpu:1 # Number of GPUs required
#SBATCH --gres=gpu:1
## SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=10 # Number of CPUs per task
#SBATCH --mem=40gb
#SBATCH --time 6:00:00
#SBATCH --partition=gpu-large 
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket
#SBATCH --qos=batch-short

## conda activate llmrl
# Run you sh script
# python run_gsm8k.py --model_name=Qwen/Qwen2.5-0.5B-Instruct --use_ir=knn+ --num_shots=0 --k=1 --nepochs=1 --seed 0 --bs 2 --gc 8 --L 200
CUDA_VISIBLE_DEVICES=0 python run_gsm8k.py --model_name=$1 --use_ir=$2 --num_shots=$3 --k=$4  --nepochs=$5 --seed $6 --bs $7 --gc $8 --L $9 --do_eval ${10}