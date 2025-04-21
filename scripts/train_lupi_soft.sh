#!/bin/bash
# export PYTHONPATH="${PYTHONPATH}:/home/gzr/code/BLEND"
DEVICE_ID=3
log_dir='train_logs/'

exp_tag='mc_maze_LUPI_soft'

# distill_type='feature'
# distill_type='correlation'
distill_type='soft'

model='NDT-LUPI'
# model='LFADS-LUPI'

fp='FP'
# fp='noFP'

echo "
Experiment Name: $exp_tag
Model: $model
Distill Type: $distill_type
Using GPU No.: $DEVICE_ID
Foward Prediction: $fp
"
CUDA_VISIBLE_DEVICES=$DEVICE_ID nohup python -u src/run_LUPI.py --run-type train --exp-config configs/$exp_tag.yaml > $log_dir$exp_tag"_model_"$model"_"$fp"_"$distill_type"_distill.log" 2>&1 &