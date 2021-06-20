#!/usr/bin/env bash

srun --reservation=fri --constraint=gpu ./main_gpu ${1:-"../imgs/input/bear_8k.jpg"} -o ${2:-"../imgs/output/result.jpg"}
