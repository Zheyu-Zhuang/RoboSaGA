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
@click.option("-d", "--device", default="cuda:0")
@click.option("-c", "--checkpoint", required=True)
@click.option("-n", "--n_eval_rollouts", default=50)
@click.option("-o", "--output_dir", required=True)
# Add options for visual domain shifts
@click.option("--lighting_and_shadow", is_flag=True, default=False)
@click.option("--lighting", is_flag=True, default=False)
@click.option("--distractors", is_flag=True, default=False)
@click.option("--backgrounds", is_flag=True, default=False)
def main(
    checkpoint,
    output_dir,
    device,
    n_eval_rollouts,
    lighting_and_shadow,
    lighting,
    distractors,
    backgrounds,
):
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
    assert not (
        lighting_and_shadow and lighting
    ), "Cannot have both lighting and shadow and lighting"

    lighting_mode = "default"
    if lighting:
        lighting_mode = "lighting"
    elif lighting_and_shadow:
        lighting_mode = "lighting_and_shadow"

    with open_dict(cfg):
        cfg.task.env_runner.distractors = distractors
        cfg.task.env_runner.rand_texture = backgrounds
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
