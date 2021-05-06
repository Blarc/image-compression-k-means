#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --reservation=fri

srun ./main_serial ../imgs/input/bear_small.jpg -m 300 -k 64 -o ../imgs/output/result.jpg