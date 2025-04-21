#!/bin/bash
# export PYTHONPATH="${PYTHONPATH}:/home/gzr/code/BLEND"
DEVICE_ID=3
log_dir='train_logs/'
exp_tag='mc_maze'

# model='LFADS'
model='NDT'
fp='FP'
# fp='noFP'
echo "
Experiment Name: $exp_tag
Model: $model
Using GPU No.: $DEVICE_ID
Foward Prediction: $fp
"
CUDA_VISIBLE_DEVICES=$DEVICE_ID nohup python -u src/run.py --run-type train --exp-config configs/$exp_tag.yaml > $log_dir$exp_tag"_model_"$model"_"$fp".log" 2>&1 &
