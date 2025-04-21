# from nlb_tools.nwb_interface import NWBDataset
from nwb_interface import NWBDataset
# from nlb_tools.make_tensors import (
#     make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
# )
from make_tensors import (
    make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
)
# from nlb_tools.evaluation import evaluate
from evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import os
import os.path as osp
from pathlib import Path
import sys

# Add ndt src if not in path
module_path = osp.abspath(osp.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as f
from torch.utils import data
# from ray import tune

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from analyze_utils import init_by_ckpt

import argparse
import logging
logging.basicConfig(level=logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser(description="script to evaluate trained model")
    parser.add_argument("--exp-tag", required=True, help="Name tag of the experiment")
    parser.add_argument("--ckpt", required=True, help="Directory that stores your model checkpoint")
    parser.add_argument("--exp-config", type=str, required=True, help="path to config yaml containing info about experiment")
    parser.add_argument("--is-ray", type=bool, default=False)
    parser.add_argument("--bin-width", type=int, default=5, help="Bin width used when resample dataset")
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default="eval",
        help="run type of the experiment (train or eval)",
    )
    return parser

parser = get_parser()
args = parser.parse_args()

variant = args.exp_tag
dataset_name = variant

# variant = "mc_maze_small_from_scratch"
is_ray = args.is_ray
# is_ray = False
phase = 'val'

print(args.exp_tag)


if args.exp_tag == 'mc_maze' or args.exp_tag == 'mc_maze_LUPI':
    datapath = '/data/gzr/CNeuro/NLB21/MC_Maze_standard/sub-Jenkins/'
else:
    raise ValueError("dataset not implemented")
print("datapath:", datapath)
dataset = NWBDataset(datapath)
dataset.resample(args.bin_width)
suffix = '' if (args.bin_width == 5) else f'_{int(args.bin_width)}'

config, ckpt_path = prepare_config(args.exp_config, args.run_type, ckpt_path="", opts=None, suffix=suffix)

if is_ray:
    # tune_dir = f"/home/joelye/user_data/nlb/ndt_runs/ray/{variant}"
    # df = tune.ExperimentAnalysis(tune_dir).dataframe()
    # # ckpt_path = f"/home/joelye/user_data/nlb/ndt_runs/ray/{variant}/best_model/ckpts/{variant}.lve.pth"
    # ckpt_dir = df.loc[df["best_masked_loss"].idxmin()].logdir
    # ckpt_path = f"{ckpt_dir}/ckpts/{variant}.lve.pth"
    # runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
    pass
else:
    # ckpt_dir = Path("/data2404/gzr/home/code/neuroscience/neural-data-transformers/ndt_runs/")
    ckpt_dir = Path(args.ckpt)
    fp = "noFP" if config.DATA.IGNORE_FORWARD else "FP"
    ckpt_path = ckpt_dir / variant / config.MODEL.NAME / fp / f"{variant}.lve.pth"
    print("ckpt_path:", ckpt_path)
    # runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
    #*
    runner, spikes, rates, behavior, heldout_spikes, forward_spikes, forward_behavior = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
    #*

eval_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.val
)
train_rates, _ = runner.get_rates(
    checkpoint_path=ckpt_path,
    save_path = None,
    mode = DATASET_MODES.train
)

# * Val
eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

output_dict = {
    dataset_name + suffix: {
        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
        'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
        # 'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
        # 'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
    }
}

# Reset logging level to hide excessive info messages
logging.getLogger().setLevel(logging.WARNING)

# If 'val' phase, make the target data
if phase == 'val':
    # Note that the RTT task is not well suited to trial averaging, so PSTHs are not made for it
    target_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', include_psth=True, save_file=False)

    # Demonstrate target_dict structure
    print(target_dict.keys())
    print(target_dict[dataset_name + suffix].keys())

# Set logging level again
logging.getLogger().setLevel(logging.INFO)

if phase == 'val':
    print(evaluate(target_dict, output_dict))

# e.g. with targets to compare to
# target_dict = torch.load(f'/snel/home/joely/tmp/{variant}_target.pth')
# target_dict = np.load(f'/snel/home/joely/tmp/{variant}_target.npy', allow_pickle=True).item()

# print(evaluate(target_dict, output_dict))

# e.g. to upload to EvalAI
# with h5py.File('ndt_maze_preds.h5', 'w') as f:
#     group = f.create_group('mc_maze')
#     for key in output_dict['mc_maze']:
#         group.create_dataset(key, data=output_dict['mc_maze'][key])