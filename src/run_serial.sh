#!/usr/bin/env bash

srun --reservation=fri --ntasks=1 ./main_serial ${1:-"../imgs/input/bear_medium.jpg"} -o ${2:-"../imgs/output/result.jpg"}