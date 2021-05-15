#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --reservation=fri

srun ./main_serial ../imgs/input/bear_small.jpg -o ../imgs/output/result.jpg