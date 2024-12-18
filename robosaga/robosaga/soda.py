import os
import random

import torch
from PIL import Image
from torchvision import transforms

try:
    from robomimic.models.obs_core import VisualCore
except ImportError:
    from robomimic.models.base_nets import VisualCore

import torch.nn.functional as F
from torch import nn


class SODA:
    """
    A Wrapper of SODA (Self-Supervised Object Detection and Augmentation) for RoboMimic Visual Core
    """
    def __init__(
        self, encoder, ema_encoder, projection, ema_projection, alpha=0.5, **kwargs
    ):
        self.encoder = encoder
        self.ema_encoder = ema_encoder
        self.proj = projection
        self.ema_proj = ema_projection
        self.background_images = self.preload_all_backgrounds(kwargs["background_path"])
        self.normalizer = kwargs.get("normalizer", None)
        # augmentation index fixed across obs pairs
        self.alpha = alpha
        self.loss = nn.MSELoss()
        params = list(self.encoder.parameters()) + list(self.proj.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-4)
        self.epoch_idx = 0  # epoch index
        self.batch_idx = 0  # batch index

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    def ema_update(self, m=0.995):
        with torch.no_grad():
            for p, ema_p in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
                ema_p.data.mul_(m).add_((1 - m) * p.data)
            for p, ema_p in zip(self.proj.parameters(), self.ema_proj.parameters()):
                ema_p.data.mul_(m).add_((1 - m) * p.data)

    # the main function to be called for data augmentation
    def step_train_epoch(self, obs_dict, epoch_idx, batch_idx, validate=False):
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict)
        vec_ema = []
        vec = []
        for i, obs_key in enumerate(obs_meta["visual_modalities"]):
            im = obs_dict[obs_key]
            rand_bg_idx = random.sample(range(self.background_images.shape[0]), len(im))
            bg = obs_meta["randomizers"][i].forward_in(self.background_images[rand_bg_idx])
            bg = self.normalizer[obs_key].normalize(bg) if self.normalizer is not None else bg
            aug_im = im * self.alpha + bg * (1 - self.alpha)
            vec.append(self.encoder.obs_nets[obs_key](aug_im))
            with torch.no_grad():
                vec_ema.append(self.ema_encoder.obs_nets[obs_key](im))
        with torch.no_grad():
            proj_vec_ema = self.ema_proj(torch.cat(vec_ema, dim=1))
            proj_vec_ema = F.normalize(proj_vec_ema, p=2, dim=1)
        proj_vec = self.proj(torch.cat(vec, dim=1))
        proj_vec = F.normalize(proj_vec, p=2, dim=1)
        if not validate:
            loss = self.loss(proj_vec_ema, proj_vec)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_update()
        return self.restore_obs_dict_shape(obs_dict, obs_meta)

    def prepare_obs_dict(self, obs_dict):
        # get visual modalities and randomizers
        visual_modalities = [
            k for k, v in self.encoder.obs_nets.items() if isinstance(v, VisualCore)
        ]
        randomizers = [self.encoder.obs_randomizers[k] for k in visual_modalities]

        vis_obs_dim = obs_dict[visual_modalities[0]].shape
        has_temporal_dim = len(vis_obs_dim) > 4
        n_samples = vis_obs_dim[0] if not has_temporal_dim else vis_obs_dim[0] * vis_obs_dim[1]

        obs_meta = {
            "temporal_dim": vis_obs_dim[1] if has_temporal_dim else 0,
            "visual_modalities": visual_modalities,
            "n_samples": n_samples,
            "randomizers": randomizers,
        }

        def flatten_temporal_dim(x):
            raw_dim = x.shape
            return (x.view(n_samples, *raw_dim[2:]), raw_dim) if has_temporal_dim else (x, raw_dim)

        for k in self.encoder.obs_shapes.keys():
            obs_dict[k], raw_shape = flatten_temporal_dim(obs_dict[k])
            obs_meta[k] = {
                "raw_shape": raw_shape,
                "is_visual": k in visual_modalities,
                "input_shape": list(raw_shape),
            }

        for vis_obs, randomizer in zip(visual_modalities, randomizers):
            x, crop_inds = randomizer.forward_in(obs_dict[vis_obs], return_inds=True)
            obs_meta[vis_obs]["input_shape"][-3:] = x.shape[-3:]
            obs_meta[vis_obs]["crop_inds"] = crop_inds
            obs_dict[vis_obs] = x

        return obs_dict, obs_meta

    @staticmethod
    def restore_obs_dict_shape(obs_dict, obs_meta):
        for k in obs_dict.keys():
            obs_dict[k] = obs_dict[k].view(obs_meta[k]["input_shape"])
        return obs_dict

    @staticmethod
    def preload_all_backgrounds(background_path, im_size=(84, 84)):
        all_f_names = os.listdir(background_path)
        all_im_names = []
        for f_name in all_f_names:
            if f_name.endswith(".jpg") or f_name.endswith(".png"):
                all_im_names.append(f_name)
        n_images = len(all_im_names)
        backgrounds = torch.zeros((n_images, 3, im_size[0], im_size[1]))
        for i, im_name in enumerate(all_im_names):
            im = Image.open(os.path.join(background_path, im_name))
            if im.size != im_size:
                im = im.resize(im_size)
            im = transforms.ToTensor()(im)
            backgrounds[i] = im
        return backgrounds.to("cuda")
