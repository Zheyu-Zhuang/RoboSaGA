"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib

import click
import dill
import hydra
import torch
import wandb
from omegaconf.omegaconf import open_dict

from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-l", "lighting_mode", default="default")
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--distractors", is_flag=True, default=False) 
@click.option("--rand_texture", is_flag=True, default=False)

def main(checkpoint, output_dir, device, distractors, rand_texture, lighting_mode):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    n_eval_rollouts = 50
    with open_dict(cfg):
        cfg.task.env_runner.distractors = distractors
        cfg.task.env_runner.rand_texture = rand_texture
        cfg.task.env_runner.lighting_mode = lighting_mode
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_test = n_eval_rollouts
        cfg.task.env_runner.n_envs = n_eval_rollouts
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print(json_log)


if __name__ == "__main__":
    main()