#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --export=ALL
#SBATCH --job-name="gputut"
#SBATCH --nodes=1
#SBATCH --output="gputut.%j.%N.out"
#SBATCH -t 00:30:00
##################### SLURM (do not change) ^  #####################



module load cmake
module load gcc/10.3.0
module load rocm/4.2.0



######### NOTE: run this script with sbatch command #############


### set it to to run you compiled code on the compute nodes.
BINPATH=./build/
LOGS="./build/logs/"
#LOGS="./build/logs-${DATE}/"


mkdir $LOGS

$BINPATH/vec_mul_vec   --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/vec_add.csv

###  plotting: add your plots here if needed
