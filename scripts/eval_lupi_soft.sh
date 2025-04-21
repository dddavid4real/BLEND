# export PYTHONPATH="${PYTHONPATH}:/home/gzr/code/BLEND"
log_dir='eval_logs/'
DEVICE_ID=3

exp_tag='mc_maze_LUPI_soft'

# distill_type='feature'
# distill_type='correlation'
distill_type='soft'

model='NDT-LUPI'
# model='LFADS-LUPI'

# fp='noFP'
fp='FP'

echo "
Experiment Name: $exp_tag
Model: $model
Distill Type: $distill_type
Using GPU No.: $DEVICE_ID
Foward Prediction: $fp
"

CUDA_VISIBLE_DEVICES=$DEVICE_ID nohup python -u scripts/eval_lupi_soft.py \
    --exp-tag $exp_tag \
    --exp-config configs/$exp_tag.yaml \
    --ckpt 'ndt_runs/' > $log_dir$exp_tag"_model_"$model"_"$fp"_"$distill_type"_distill.log" 2>&1 &