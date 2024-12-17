#!/bin/bash

python3 main.py \
    --learning_rate 0.002 \
    --num_epochs 20 \
    --num_clients_per_run 256 \
    --device cuda \
    --dataset adult \
    --warmup_epochs 20 \
    --warmup_type inclusive \
    --target_label 1