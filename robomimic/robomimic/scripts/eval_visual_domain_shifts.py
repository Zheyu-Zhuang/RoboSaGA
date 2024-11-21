import concurrent.futures
import os
import re
import shutil
import subprocess

import numpy as np

from robomimic.utils.eval_utils import get_top_n_experiments


def run_script(script_name, script_args):
    command = ["python", script_name] + script_args
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate()
    return script_name, script_args, output, errors


def run_scripts_in_parallel(scripts_with_args, output_file):

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_script, script_name, script_args)
            for script_name, script_args in scripts_with_args
        ]

    with open(output_file, "a") as f:
        for future in concurrent.futures.as_completed(futures):
            script_name, script_args, output, errors = future.result()
            f.write(
                f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n"
            )
            print(f"Output from {script_name} with arguments {' '.join(script_args)}:\n{output}\n")
        if errors:
            f.write(f"Errors: {errors}\n")
            print(f"Errors: {errors}\n")


def extract_success_rates(file_path):
    pattern = r'"Success_Rate": ([\d\.]+)'
    success_rates = []
    with open(file_path, "r") as file:
        content = file.read()
        matches = re.findall(pattern, content)
        for match in matches:
            success_rates.append(float(match))
    return success_rates


def get_results_string(output_file, top_n_success_rate, offdomain_type):
    success_rates = extract_success_rates(output_file)
    indomain_ssr_mean = np.around(np.mean(top_n_success_rate), 2)
    indomain_ssr_std = np.around(np.std(top_n_success_rate), 2)
    offdomain_ssr_mean = np.around(np.mean(success_rates), 2)
    offdomain_ssr_std = np.around(np.std(success_rates), 2)

    results_string = (
        f"\n===== Results for {args.exp_path} =====\n"
        f"Indomain: {top_n_success_rate}, {indomain_ssr_mean} +/- {indomain_ssr_std}\n"
        f"  data: {top_n_success_rate}\n"
        f"Off-domain for {offdomain_type}: {offdomain_ssr_mean} +/- {offdomain_ssr_std}\n"
        f"   data: {success_rates}\n"
    )

    return results_string


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_path", type=str, default=None, required=True, help="Path to experiment folder"
    )
    parser.add_argument("--video", action="store_true", help="Save video of evaluation")
    parser.add_argument(
        "--vds",
        nargs="+",
        default=[],
        help="Visual domain shifts to evaluate, supported: backgrounds, distractors, lighting, lighting_and_shadow",
    )
    parser.add_argument(
        "--top_n", type=int, default=3, help="Number of top checkpoints to evaluate"
    )
    parser.add_argument(
        "--n_rollouts", type=int, default=50, help="Number of rollouts per checkpoint"
    )
    parser.add_argument(
        "--empty_scene_temp_files", action="store_true", help="Remove temp xml files"
    )
    args = parser.parse_args()

    this_file_path = os.path.abspath(__file__)
    robostuie_dir = os.path.join(os.path.dirname(this_file_path), "../../../robosuite")
    asset_path = os.path.join(robostuie_dir, "robosuite/models/assets/arenas")

    if args.empty_scene_temp_files:
        all_files = os.listdir(asset_path)
        print(all_files)
        for f in all_files:
            f_name = f.split(".")[0]
            if "temp" in f_name.split("_"):
                os.remove(os.path.join(asset_path, f))
                print(f"Removed {f}")
        exit()

    for vds in args.vds:
        if vds not in ["backgrounds", "distractors", "lighting", "lighting_and_shadow"]:
            raise ValueError(f"Invalid visual domain shift type: {vds}")

    if "lighting" in args.vds and "lighting_and_shadow" in args.vds:
        args.vds.remove("lighting")  # lighting_and_shadow already includes lighting

    log_file_path = os.path.join(args.exp_path, "logs/log.txt")
    eval_dir = os.path.join(args.exp_path, "eval")
    video_dir = os.path.join(eval_dir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    top_n_checkpoints, top_n_success_rate = get_top_n_experiments(log_file_path, n=args.top_n)
    # in case save path changes
    top_n_checkpoints = [os.path.basename(ckpt) for ckpt in top_n_checkpoints]
    top_n_checkpoints = [os.path.join(args.exp_path, "models", ckpt) for ckpt in top_n_checkpoints]

    py_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "eval_trained_agent.py")
    scripts_with_args = []

    # TODO: add combinations of visual domain shifts

    mode = "_".join(args.vds) if args.vds else "train_domain"
    vds_command = [f"--{vds}" for vds in args.vds]

    print("\n=====================")
    print(f"Running evaluation for {mode} with top {args.top_n} checkpoints")

    for i, ckpt_path in enumerate(top_n_checkpoints):
        ckpt_name = os.path.basename(ckpt_path).replace(".pth", "")
        video_name = f"{mode}_ckpt_{ckpt_name}.mp4"
        video_path = os.path.join(video_dir, video_name)
        video_command = ["--video_path", video_path] if args.video else []
        rand_id = np.random.randint(10000000)
        scripts_with_args.append(
            (
                py_script,
                [
                    "--agent",
                    ckpt_path,
                    "--n_rollouts",
                    str(args.n_rollouts),
                    "--env_id",
                    f"{mode}_env_{rand_id}",
                ]
                + vds_command
                + video_command,
            )
        )
    output_file = os.path.join(eval_dir, f"{mode}_stats.txt")
    # Execute each script with its arguments and save the output
    if os.path.exists(output_file):
        archive_folder = os.path.join(eval_dir, "archive")
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)
        n_old = len(os.listdir(archive_folder))
        shutil.move(output_file, os.path.join(archive_folder, f"{mode}_stats_{n_old}.txt"))
        print("WARNING: output file already exists, moving to archive folder")
    run_scripts_in_parallel(scripts_with_args, output_file)
    stats = get_results_string(output_file, top_n_success_rate, mode)
    print(stats)

    with open(output_file, "a") as f:
        f.write(stats)
