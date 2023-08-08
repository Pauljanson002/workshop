#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J batch_job
#SBATCH -o /ibex/ai/home/jansonrp/workshop/tmp/%J.out
#SBATCH -e /ibex/ai/home/jansonrp/workshop/tmp/%J.err
#SBATCH --time=01-01:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --reservation=A100

source ~/miniconda3/bin/activate /ibex/ai/home/jansonrp/workshop/env

# python /ibex/ai/home/jansonrp/workshop/imagnet100.py > imagenet100.txt

# python /ibex/ai/home/jansonrp/workshop/imagenet1000.py > imagenet1000_1.txt

# python /ibex/ai/home/jansonrp/workshop/imagenet21k.py > imagenet21k_1.txt

#python /ibex/ai/home/jansonrp/workshop/imagenet-r.py > imagenet-r_11111.txt
python /ibex/ai/home/jansonrp/workshop/main_core50.py > core50_1.txt