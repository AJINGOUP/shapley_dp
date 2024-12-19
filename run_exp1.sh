#!/bin/bash

python3 main.py \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --num_clients_per_round 128 \
    --max_rounds 8 \
    --device cuda:1 \
    --dataset medmnist \
    --seed 666