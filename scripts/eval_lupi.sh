# export PYTHONPATH="${PYTHONPATH}:/home/gzr/code/BLEND"
log_dir='eval_logs/'
DEVICE_ID=3

exp_tag='mc_maze_LUPI'

model='NDT-LUPI'
# model='LFADS-LUPI'
# fp='noFP'
fp='FP'

echo "
Experiment Name: $exp_tag
Model: $model
Using GPU No.: $DEVICE_ID
Foward Prediction: $fp
"

CUDA_VISIBLE_DEVICES=$DEVICE_ID nohup python -u scripts/eval_lupi.py \
    --exp-tag $exp_tag \
    --exp-config configs/$exp_tag.yaml \
    --ckpt 'ndt_runs/' > $log_dir$exp_tag"_model_"$model"_"$fp".log" 2>&1 &