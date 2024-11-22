import os
import shutil

from ruamel.yaml import YAML

this_dir = os.path.dirname(os.path.abspath(__file__))
all_tasks = os.listdir(os.path.join(this_dir, "task"))
all_tasks = [os.path.basename(task) for task in all_tasks]
all_tasks = [task.split(".")[0] for task in all_tasks]

aug_types = ["baseline", "overlay", "saga", "soda", "color_jitter"]

f_base = "train_diffusion_unet_hybrid"

template_file = os.path.join(this_dir, "_" + f_base + "_task_template.yaml")

yaml = YAML()

for task in all_tasks:
    print(f"Processing task {task}")
    for aug_type in aug_types:
        out_file = os.path.join(this_dir, f"{f_base}_{task}_{aug_type}.yaml")
        f_path = os.path.join(this_dir, template_file)
        shutil.copy(f_path, out_file)

        # Read YAML as dictionary while preserving structure
        with open(out_file, "r") as file:
            yaml_dict = yaml.load(file)

        # Modify the dictionary as needed
        yaml_dict["defaults"][1]["task"] = task
        yaml_dict["exp_name"] = aug_type
        if aug_type == "color_jitter":
            yaml_dict["color_jitter"] = True
        elif aug_type == "overlay":
            yaml_dict["saliency"]["enabled"] = True
            yaml_dict["saliency"]["aug_strategy"] = "overlay"
        elif aug_type == "saga":
            yaml_dict["group_norm"]["return_fullgrad_bias"] = True
            yaml_dict["saliency"]["enabled"] = True
        elif aug_type == "soda":
            yaml_dict["policy"][
                "_target_"
            ] = "diffusion_policy.policy.diffusion_unet_hybrid_image_policy_soda.DiffusionUnetHybridImagePolicySODA"
        # Write the updated dictionary back to the YAML file while preserving structure
        with open(out_file, "w") as file:
            yaml.dump(yaml_dict, file)
