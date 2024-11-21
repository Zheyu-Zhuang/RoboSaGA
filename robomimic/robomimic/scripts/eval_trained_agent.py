"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import h5py
import imageio
import numpy as np
import torch
from tqdm import tqdm

import diffusion_policy.model.vision.crop_randomizer as dmvc
import robomimic.models.base_nets as rmbn
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robosuite
from diffusion_policy.common.pytorch_util import replace_submodules
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.0
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(
                                mode="rgb_array", height=512, width=512, camera_name=cam_name
                            )
                        )
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def replace_texture(xml_file, wall_textures, table_textures, floor_textures):
    # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    table_env_target_texture = ["tex-ceramic", "tex-cream-plaster", "texplane"]
    multi_table_target_texture = ["tex-ceramic", "tex-cream-plaster", "texplane"]
    bin_env_target_texture = ["tex-light-wood", "tex-dark-wood", "texplane", "tex-ceramic", "tex-cream-plaster"]
    texture_types = {
        "tex-ceramic": table_textures,
        "tex-cream-plaster": wall_textures,
        "texplane": floor_textures,
        "tex-light-wood": floor_textures,
        "tex-dark-wood": floor_textures,
    }
    env_name = os.path.basename(xml_file).split(".")[0]
    if env_name.endswith("_temp"):
        env_name = env_name.split("_")[:2]
        env_name = "_".join(env_name)
    for texture in root.iter("texture"):
        attrib_name = texture.attrib.get("name")
        if env_name == "pegs_arena" or "table_arena" in env_name:
            if attrib_name in table_env_target_texture:
                texture.attrib["file"] = random.choice(texture_types[attrib_name])
        elif env_name == "multi_table":
            if attrib_name in multi_table_target_texture:
                texture.attrib["file"] = random.choice(texture_types[attrib_name])
        elif env_name == "bins_arena":
            if attrib_name in bin_env_target_texture:
                texture.attrib["file"] = random.choice(texture_types[attrib_name])
    tree.write(xml_file)


def get_robosuite_path():
    this_file_path = os.path.abspath(__file__)
    return os.path.join(os.path.dirname(this_file_path), "../../../robosuite")


def get_all_texture_paths(rand_texture):
    if rand_texture is None:
        return None
    robosuite_path = get_robosuite_path()
    texture_dir = os.path.join(
        robosuite_path, "robosuite/models/assets/textures/evaluation_textures"
    )
    texture_dir = os.path.join(texture_dir, rand_texture)
    texture_paths = []
    for texture_file in os.listdir(texture_dir):
        texture_path = os.path.join(texture_dir, texture_file)
        texture_paths.append(texture_path)
    return texture_paths

def randomize_lighting(mode=None, xml_file=None):
    if mode is None or xml_file is None:
        return
    assert mode in ['lighting', 'lighting_and_shadow', None]
    assert xml_file is not None
    tree = ET.parse(xml_file)
    root = tree.getroot()
    intensity_range = [0.1, 0.8]
    for light in root.iter("light"):
        low, high = intensity_range
        r, g, b = random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)
        light.attrib["diffuse"] = f"{r} {g} {b}"
        if 'shadow' in mode.split('_'):
            light.attrib["castshadow"] = "true"
            light.attrib["diffuse"] = f"{r} {g} {b}"
    tree.write(xml_file)
    
floor_textures = get_all_texture_paths("floor")
wall_textures = get_all_texture_paths("wall")
table_textures = get_all_texture_paths("table")


def run_trained_agent(args):
    # some arg checking
    write_video = args.video_path is not None
    assert not (args.render and write_video)  # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=False
    )

    # center cropping for eval
    replace_submodules(
        root_module=policy.policy.nets["policy"].nets["encoder"].nets["obs"],
        predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
        func=lambda x: dmvc.CropRandomizer(
            input_shape=x.input_shape,
            crop_height=x.crop_height,
            crop_width=x.crop_width,
            num_crops=x.num_crops,
            pos_enc=x.pos_enc,
        ),
    )

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    if args.distractors:
        args.env_id = None
        
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=False,
        distractors=args.distractors,
        env_id=args.env_id,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = args.dataset_path is not None
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    pbar = tqdm(total=rollout_num_episodes)

    xml_path = env.env.xml

    for i in range(pbar.total):
        if args.backgrounds:
            replace_texture(xml_path, wall_textures, table_textures, floor_textures)
        if args.lighting: 
            randomize_lighting(mode='lighting', xml_file=xml_path)
        if args.lighting_and_shadow:
            randomize_lighting(mode='lighting_and_shadow', xml_file=xml_path)
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)
        rollout_stats_temp = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        success_rate = np.sum(rollout_stats_temp["Success_Rate"])
        pbar.set_postfix({"Success Rate": success_rate / len(rollout_stats_temp["Success_Rate"])})
        pbar.update()

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(
                        "next_obs/{}".format(k), data=np.array(traj["next_obs"][k])
                    )

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                    "model"
                ]  # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                0
            ]  # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print(f"Average Rollout Stats for {ckpt_path}:")
    print(json.dumps(avg_rollout_stats, indent=4))
    
    if 'temp' in xml_path:
        print(f"Removing temporary xml file {xml_path}")
        os.remove(xml_path)

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)  # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    parser.add_argument(
        "--distractors",
        action="store_true",
        help="use distractors in the environment",
    )
    
    parser.add_argument(
        "--lighting",  
        action="store_true",
        help="randomize lighting in the environment",
    )
    
    parser.add_argument(
        "--lighting_and_shadow",
        action="store_true",
        help="randomize lighting and shadows in the environment",
    )

    parser.add_argument("--backgrounds", action="store_true")
    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action="store_true",
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument("--env_id", type=str, default="1")

    args = parser.parse_args()
    run_trained_agent(args)