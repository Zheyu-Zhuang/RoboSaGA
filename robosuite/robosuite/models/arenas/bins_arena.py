import os
import shutil

import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion
from robosuite.utils.saga_utils import (
    get_all_texture_paths,
    randomize_lighting,
    replace_texture,
)


class BinsArena(Arena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        bin1_pos (3-tuple): (x,y,z) position to place bin1
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self,
        bin1_pos=(0.1, -0.5, 0.8),
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        rand_texture=False,
        lighting_mode="default",
        env_id=None,
        xml="arenas/bins_arena.xml",
    ):
        default_xml = xml_path_completion(xml)
        if env_id is not None:
            xml_temp = default_xml.replace(".xml", f"_{env_id}_temp.xml")
            if not os.path.exists(xml_temp):
                shutil.copy(default_xml, xml_temp)
            xml = xml_temp
        else:
            xml = default_xml

        if lighting_mode == "random":
            randomize_lighting(xml)
        elif lighting_mode == "shadow":
            randomize_lighting(xml, castshadow=True)

        if rand_texture is True:
            floor_textures = get_all_texture_paths("floor")
            wall_textures = get_all_texture_paths("wall")
            table_textures = get_all_texture_paths("table")
            replace_texture(xml_temp, wall_textures, table_textures, floor_textures)
            xml = xml_temp

        super().__init__(xml)

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.bin1_body = self.worldbody.find("./body[@name='bin1']")
        self.bin2_body = self.worldbody.find("./body[@name='bin2']")
        self.table_top_abs = np.array(bin1_pos)

        self.configure_location()
        self.xml = xml

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))
