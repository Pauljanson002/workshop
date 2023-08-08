#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J batch_job
#SBATCH -o /ibex/ai/home/jansonrp/workshop/tmp/%J.out
#SBATCH -e /ibex/ai/home/jansonrp/workshop/tmp/%J.err
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL

source ~/miniconda3/bin/activate /ibex/ai/home/jansonrp/workshop/env

python /ibex/ai/home/jansonrp/workshop/main_core50.py