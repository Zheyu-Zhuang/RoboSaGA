import os
import random
import shutil
import xml.etree.ElementTree as ET

from robosuite.models.objects import (
    BottleObject,
    BreadObject,
    CanObject,
    CerealObject,
    LemonObject,
    MilkObject,
)

# fix the random seed for reproducibility

random.seed(0)


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


def distractors_to_model(distractors, postfix=""):
    if distractors is None:
        return []
    supported_distractors = {
        "bottle": BottleObject,
        "lemon": LemonObject,
        "milk": MilkObject,
        "bread": BreadObject,
        "can": CanObject,
        "cereal": CerealObject,
    }
    idx = 0
    models = []
    for distractor_ in distractors:
        if distractor_ not in supported_distractors.keys():
            raise ValueError(
                "Distractor {} not supported. Supported distractors are {}".format(
                    distractor_, supported_distractors.keys()
                )
            )
        else:
            name = f"{distractor_}_{idx}_{postfix}"
            models.append(supported_distractors[distractor_](name=name))
            idx += 1
    return models


def randomize_lighting(mode=None, xml_file=None):
    if mode is None or xml_file is None:
        return
    assert mode in ["lighting", "lighting_and_shadow", None]
    assert xml_file is not None
    tree = ET.parse(xml_file)
    root = tree.getroot()
    intensity_range = [0.1, 0.8]
    for light in root.iter("light"):
        low, high = intensity_range
        r, g, b = random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)
        light.attrib["diffuse"] = f"{r} {g} {b}"
        if "shadow" in mode.split("_"):
            light.attrib["castshadow"] = "true"
            light.attrib["diffuse"] = f"{r} {g} {b}"
    tree.write(xml_file)


def replace_texture(xml_file, wall_textures, table_textures, floor_textures):
    # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    table_env_target_texture = ["tex-ceramic", "tex-cream-plaster", "texplane"]
    multi_table_target_texture = ["tex-ceramic", "tex-cream-plaster", "texplane"]
    bin_env_target_texture = [
        "tex-light-wood",
        "tex-dark-wood",
        "texplane",
        "tex-ceramic",
        "tex-cream-plaster",
    ]
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


def get_texture_name(texture_path):
    texture_file_name = os.path.basename(texture_path)
    return texture_file_name.split(".")[0]
