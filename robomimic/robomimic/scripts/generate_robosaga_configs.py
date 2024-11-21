import copy
import json
import os
from collections import OrderedDict

# TODO: add randomize jittering as another augmentation strategy


def generate_saga_configs(config_dir, task, background_dir, data_dir, exp_dir):
    saga_default_configs = {
        "saliency": {
            "enabled": True,
            "warmup_epochs": 10,
            "update_ratio": 0.1,
            "aug_ratio": 0.5,
            "debug_vis": False,
            "debug_save": True,
            "buffer_shape": [84, 84],
            "save_dir": "",
            "save_debug_im_every_n_batches": 100,
            "background_path": background_dir,
            "aug_strategy": "robosaga",
            "saliency_erase_threshold": 0.5,  # threshold for erasing saliency
            "blend_alpha": 0.5,  # blending alpha for blending random overlay
            "saliency_saga_cap": 0.8,  # the cap for saliency values in RoboSaGA
        },
        "train": {"color_jitter": True},
    }

    # HACK: overload the saliency configs for soda for only the background path
    soda_default_configs = {
        "saliency": {"enabled": True, "background_path": background_dir},
        "train": {"color_jitter": True},
    }

    overlay_default_configs = {
        "saliency": {
            "enabled": True,
            "warmup_epochs": 10,
            "aug_strategy": "random_overlay",
            "aug_ratio": 0.5,
            "debug_vis": False,
            "debug_save": True,
            "save_dir": "",
            "save_debug_im_every_n_batches": 100,
            "background_path": background_dir,
            "saliency_erase_threshold": 0.5,  # threshold for erasing saliency
            "blend_alpha": 0.5,  # blending alpha for blending random overlay
            "saliency_saga_cap": 0.8,  # the cap for saliency values in RoboSaGA
        },
        "train": {"color_jitter": True},
    }

    saga_exp_configs = {
        "saga": saga_default_configs,
        "soda": soda_default_configs,
        "overlay": overlay_default_configs,
        "baseline": {},
        "baseline_jitter": {"train": {"color_jitter": True}},
    }

    input_dir = os.path.join(config_dir, "templates", f"{task}")
    output_dir = os.path.join(config_dir, f"{task}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for net_type in ["bc", "bc_rnn"]:
        input_file_path = os.path.join(input_dir, f"{net_type}.json")
        assert os.path.exists(input_file_path), f"File {input_file_path} does not exist"
        with open(input_file_path, "r") as f:
            print(f"Reading config file: {input_file_path}")
            task_configs = json.load(f)
            for key, value in saga_exp_configs.items():
                #
                config_temp = copy.deepcopy(task_configs)
                config_temp["observation"]["encoder"]["rgb"]["core_kwargs"]["backbone_kwargs"][
                    "pretrained"
                ] = "true"
                config_temp["observation"]["encoder"]["rgb"]["core_kwargs"][
                    "pool_class"
                ] = "SpatialMeanPool"
                crop_height = config_temp["observation"]["encoder"]["rgb"][
                    "obs_randomizer_kwargs"
                ]["crop_height"]
                crop_width = config_temp["observation"]["encoder"]["rgb"]["obs_randomizer_kwargs"][
                    "crop_width"
                ]
                if "saliency" in value:
                    config_temp["saliency"] = value["saliency"]
                    config_temp["saliency"]["output_shape"] = [crop_height, crop_width]
                use_jitter = value.get("train", {}).get("color_jitter", False)
                config_temp["train"]["color_jitter"] = use_jitter
                config_temp["train"]["data"] = os.path.join(data_dir, f"{task}/ph/image.hdf5")
                config_temp["train"]["output_dir"] = os.path.join(
                    exp_dir, f"{task}_image/{net_type}"
                )
                config_temp["experiment"]["name"] = f"{key}"
                if net_type == "bc":
                    config_temp["train"]["batch_size"] = 64
                if task == "lift":
                    config_temp["train"]["num_epochs"] = 200
                config_temp["train"]["num_data_workers"] = 8

                output_file_path = os.path.join(output_dir, f"{net_type}_{key}.json")
                with open(output_file_path, "w") as f:
                    ordered_config = OrderedDict()
                    if "saliency" in config_temp:
                        ordered_config["saliency"] = config_temp.pop("saliency")
                    for k, v in config_temp.items():
                        ordered_config[k] = v
                    json.dump(ordered_config, f, indent=4)
                print(f"Generated config file: {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the default configs",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["lift", "square", "transport", "can"]
    )
    parser.add_argument(
        "-c",
        "--config_dir",
        type=str,
        default="./robomimic/robosaga_configs",
        help="Directory containing the config templates",
    )
    parser.add_argument(
        "-b",
        "--background_dir",
        type=str,
        default="../data/backgrounds",
        help="Directory containing the background images",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="../data/robomimic",
        help="Directory containing the data",
    )
    parser.add_argument(
        "-e",
        "--exp_dir",
        type=str,
        default="../experiments",
        help="Directory to save the experiments",
    )

    args = parser.parse_args()

    assert os.path.exists(args.config_dir), f"Config directory {args.config_dir} does not exist"
    assert os.path.exists(
        args.background_dir
    ), f"Background directory {args.background_dir} does not exist"
    assert os.path.exists(args.data_dir), f"Data directory {args.data_dir} does not exist"

    for task in args.tasks:
        generate_saga_configs(
            args.config_dir, task, args.background_dir, args.data_dir, args.exp_dir
        )
