#!/usr/bin/env bash

threads=${1:-2}

# Multithreading
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$threads

srun --reservation=fri --ntasks=1 --cpus-per-task=$threads ./main_omp ${2:-"../imgs/input/bear_medium.jpg"} -o ${3:-"../imgs/output/result.jpg"} -t $threads