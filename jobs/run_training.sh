#!/bin/bash

hours=$1
echo "Job scheduled to start $(date -d "+$hours hours")."
sleep $((hours * 3600))
date

cd ..
source ~/.python_venv/bin/activate
python3 train_transformer_unsupervised.py
python3 train_transformer_gridsearch.py
#python3 train_transformer.py
#python3 train_transformer_random_init.py
