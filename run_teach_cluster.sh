#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --job-name="vectut"
#SBATCH --nodes=1
#SBATCH --output="multithreadtut.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################


module load TeachEnv/2022a
module load gcc/13.2.0
module load cmake


######### NOTE: run this script with sbatch command #############


### set it to to run you compiled code on the compute nodes.
BINPATH=./build/
LOGS="./build/logs/"
#LOGS="./build/logs-${DATE}/"


mkdir $LOGS

$BINPATH/vec_mul_vec   --benchmark_perf_counters="L1-dcache-load-misses,L1-dcache-loads" --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/vec_add.csv

###  plotting: add your plots here if needed