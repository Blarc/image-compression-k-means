#!/usr/bin/env bash

srun --reservation=fri --time=5 --ntasks=1 --cpus-per-task=1 ./main_serial ${1:-"../imgs/input/bear_small.jpg"} -o ${2:-"../imgs/output/result_serial.jpg"}