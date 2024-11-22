import os
import random

import cv2
import numpy as np
import torch

try:
    from robomimic.models.obs_core import VisualCore
except ImportError:
    from robomimic.models.base_nets import VisualCore

import robosaga.vis_utils as vis_utils
from robosaga.background_randomizer import BackgroundRandomizer
from robosaga.buffer_manager import BufferManager
from robosaga.fullgrad import FullGrad
from robosaga.tensor_extractors import EncoderOnly


class RoboSaGA:
    def __init__(self, model, normalizer=None, **kwargs):
        # Superimposition parameters
        # Erase & Erase Overlay
        self.saliency_erase_threshold = kwargs.get("saliency_erase_threshold", 0.5)
        # Blend Alpha for Random Overlay, Erase Overlay
        self.blend_alpha = kwargs.get("blend_alpha", 0.5)
        # The cap for saliency values in RoboSaGA
        self.saliency_saga_cap = kwargs.get("saliency_saga_cap", 0.8)

        # Augmentation related attributes
        self.aug_strategy = kwargs.get("aug_strategy", "robosaga")        
        self.aug_ratio = kwargs.get("aug_ratio", 0)
        self.target_aug_ratio = kwargs.get("aug_ratio", 0)
        self.update_ratio = kwargs.get("update_ratio", 0)
        self.warmup_epochs = kwargs.get("warmup_epochs", 10)

        assert self.aug_ratio > 0, "aug_ratio should be greater than 0"

        if self.aug_strategy == "robosaga":
            assert self.update_ratio > 0, "update_ratio should be greater than 0"
        self.check_augmentation_strategy()

        self.disable_buffer = kwargs.get("disable_buffer", False)

        # Other attributes
        self.save_debug_im_every_n_batches = kwargs.get("save_debug_im_every_n_batches", 50)
        self.debug_vis = kwargs.get("debug_vis", False)
        self.debug_save = kwargs.get("debug_save", True)
        self.save_dir = kwargs.get("save_dir", None)
        self.normalizer = normalizer
        self.vis_out_dir = kwargs.get("vis_out_dir", None)
        self.output_shape = kwargs.get("output_shape", None)
        self.backgrounds = BackgroundRandomizer(**kwargs)

        self.check_required_args(print_args=True)

        # Indexes
        self.epoch_idx = 0
        self.batch_idx = 0

        # Registration status
        self.model = model
        self.is_training = True
        self.is_registered = False

        # Model and extractors
        if self.aug_strategy == "robosaga":
            self.extractors, self.buffers = self.initialize_extractors_and_buffers(**kwargs)
            self.is_registered = True

        print("=========================================================================\n")

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    def __call__(self, obs_dict, golbal_idx, epoch_idx, batch_idx):
        self.is_training = self.model.training
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        if epoch_idx < self.warmup_epochs:
            self.unregister_hooks()
            return obs_dict
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict)
        self.model.eval()  # required for saliency computation
        traj_idx = random.randint(0, obs_meta["n_trajectories"] - 1)
        vis_saliency = []
        vis_aug = []
        for obs_key in obs_meta["visual_modalities"]:
            batch = obs_dict[obs_key]
            rgb = torch.clone(batch)
            normalizer = self.get_normalizer(obs_key)
            len_traj = obs_meta["trajectory_length"]
            if self.aug_strategy == "random_overlay":
                self.random_overlay(batch, normalizer)
            if self.aug_strategy == "robosaga":
                meta = obs_meta[obs_key]
                buffer = self.buffers[obs_key]
                extractor = self.extractors[obs_key]
                self.saga(batch, extractor, buffer, golbal_idx, normalizer, meta)
                smaps_ims, aug_ims = self.visualize_trajectory(
                    rgb=rgb,
                    batch=batch,
                    buffer=buffer,
                    extractor=extractor,
                    meta=meta,
                    normalizer=normalizer,
                    len_traj=len_traj,
                    global_idx=golbal_idx,
                    traj_idx=traj_idx,
                )
                vis_saliency.append(smaps_ims)
                vis_aug.append(aug_ims)
                self.save_visualization([vis_saliency, vis_aug], epoch_idx, batch_idx, traj_idx)
        self.model.train() if self.is_training else self.model.eval()
        for k in obs_dict.keys():
            obs_dict[k] = obs_dict[k].view(obs_meta[k]["input_shape"])
        return obs_dict

    def saga(self, batch, extractor, buffer, golbal_idx, normalizer, meta):
        self.register_hooks()
        if not self.is_training:
            return batch
        update_idx = self.get_update_batch_idx(golbal_idx, buffer)
        smaps = self.extract_and_update(batch, extractor, buffer, golbal_idx, update_idx, meta)
        if self.disable_buffer:
            aug_idx = update_idx
        else:
            aug_idx, smaps = self.retrieve_smaps_from_buffer(buffer, golbal_idx, meta)
        bg = self.backgrounds(len(aug_idx), normalizer)
        batch[aug_idx] = batch[aug_idx] * smaps + bg * (1 - smaps)

    def random_overlay(self, batch, normalizer):
        self.unregister_hooks()
        if not self.is_training:
            return batch
        aug_idx = torch.arange(int(batch.shape[0] * self.aug_ratio))
        bg = self.backgrounds(len(aug_idx), normalizer)
        batch[aug_idx] = batch[aug_idx] * self.blend_alpha + bg * (1 - self.blend_alpha)

    def retrieve_smaps_from_buffer(self, buffer, golbal_idx, meta):
        aug_idx = torch.arange(int(len(golbal_idx) * self.aug_ratio))
        global_ids = golbal_idx[aug_idx]
        crop_idx = meta["crop_idx"][aug_idx]
        smaps = buffer.get(global_ids, crop_idx)
        # thresholding
        strategy_values = {
            "erase": 1,
            "erase_overlay": self.blend_alpha,
        }
        if self.aug_strategy in strategy_values:
            smaps = torch.where(
                smaps < self.saliency_erase_threshold,
                torch.tensor(0, dtype=smaps.dtype, device=smaps.device),
                torch.tensor(
                    strategy_values[self.aug_strategy],
                    dtype=smaps.dtype,
                    device=smaps.device,
                ),
            )
        elif self.aug_strategy == "robosaga":
            smaps = torch.clip(smaps, 0, self.saliency_saga_cap)
        return aug_idx, smaps

    def extract_and_update(self, batch, extractor, buffer, golbal_idx, update_idx, meta):
        if not self.is_training:
            return {}
        self.model.eval()
        ims = batch[update_idx]
        smaps = extractor(ims).detach()
        norm_smaps = vis_utils.normalize_smaps(smaps)
        if not self.disable_buffer:
            buffer.set(norm_smaps, golbal_idx[update_idx], meta["crop_idx"])
        return norm_smaps

    def get_update_batch_idx(self, golbal_idx, buffer, mode="frequency"):
        """
        determine the indices of the samples to be updated and augmented,
        default mode is frequency based, where the samples with the least updates are selected
        """
        n_samples = len(golbal_idx)
        n_augs = int(n_samples * max(self.aug_ratio, self.update_ratio))
        n_updates = min(n_augs, int(n_samples * self.update_ratio))

        if n_updates == 0:
            return torch.tensor([])
        mode = "random" if self.disable_buffer else mode
        if mode == "random":
            update_batch_idx = torch.randperm(n_samples)[:n_updates]
        elif mode == "frequency":
            update_freq = buffer.update_counter[golbal_idx]
            _, sorted_idx = torch.sort(update_freq)
            update_batch_idx = sorted_idx[:n_updates]
        return update_batch_idx

    # --------------------------------------------------------------------------- #
    #                                    Utils                                    #
    # --------------------------------------------------------------------------- #

    def visualize_trajectory(
        self,
        rgb,
        batch,
        buffer,
        extractor,
        meta,
        normalizer,
        len_traj,
        global_idx,
        traj_idx,
    ):
        save_this = self.batch_idx % self.save_debug_im_every_n_batches == 0
        if self.save_dir is None or not self.debug_save or not save_this:
            return None, None
        start, end = traj_idx * len_traj, (traj_idx + 1) * len_traj
        global_idx_ = global_idx[start:end]
        rgb_ims_ = rgb[start:end]
        aug_ims_ = batch[start:end]
        if self.is_training:
            smaps = buffer.get(global_idx_, meta["crop_idx"][start:end])
            aug_vis = vis_utils.unnormalize_image(aug_ims_, normalizer)
            aug_vis = vis_utils.batch_to_ims(aug_vis)
        else:
            smaps = extractor(rgb_ims_).detach()
            aug_vis = None
        rgb_ims = vis_utils.batch_to_ims(vis_utils.unnormalize_image(rgb_ims_, normalizer))
        rgb_smaps = vis_utils.batch_to_ims(vis_utils.normalize_smaps(smaps))
        smaps_vis = 0.5 * rgb_ims + 0.5 * rgb_smaps
        smaps_vis = np.clip(smaps_vis, 0, 255).astype(np.uint8)
        return smaps_vis, aug_vis

    def save_visualization(self, ims: list, epoch_idx, batch_idx, traj_idx, padding=10):
        ims = [vis_utils.vstack_images(x) for x in ims]
        ims = vis_utils.vstack_images(ims, padding)
        if ims is None:
            return
        title = f"Trajectory {traj_idx}"
        banner = vis_utils.get_title_banner(title, ims.shape[1])
        ims = vis_utils.vstack_images([banner, ims], padding=0)
        im_name = f"batch_{batch_idx}_saliency.jpg"
        saliency_dir = os.path.join(
            self.save_dir,
            f"epoch_{epoch_idx}",
            "train" if self.is_training else "eval",
        )
        os.makedirs(saliency_dir, exist_ok=True)
        im_path = os.path.join(saliency_dir, im_name)
        return cv2.imwrite(im_path, ims)

    def prepare_obs_dict(self, obs_dict):
        obs_encoder = self.get_obs_encoder()
        # get visual modalities and randomizers
        visual_modalities = [
            k for k, v in obs_encoder.obs_nets.items() if isinstance(v, VisualCore)
        ]
        randomizers = [obs_encoder.obs_randomizers[k] for k in visual_modalities]

        vis_obs_dim = obs_dict[visual_modalities[0]].shape
        has_temporal_dim = len(vis_obs_dim) > 4
        n_samples = vis_obs_dim[0] * (vis_obs_dim[1] if has_temporal_dim else 1)

        obs_meta = {
            "temporal_dim": vis_obs_dim[1] if has_temporal_dim else 0,
            "visual_modalities": visual_modalities,
            "trajectory_length": vis_obs_dim[1] if has_temporal_dim else 1,
            "n_trajectories": vis_obs_dim[0],
            "n_samples": n_samples,
            "randomizers": randomizers,
        }

        for k in obs_encoder.obs_shapes.keys():
            raw_shape = obs_dict[k].shape
            if has_temporal_dim:
                obs_dict[k] = obs_dict[k].view(n_samples, *raw_shape[2:])
            obs_meta[k] = {
                "raw_shape": raw_shape,
                "is_visual": k in visual_modalities,
                "input_shape": list(raw_shape),
            }

        for vis_obs, randomizer in zip(visual_modalities, randomizers):
            x, crop_idx = randomizer.forward_in(obs_dict[vis_obs], return_crop_idx=True)
            obs_meta[vis_obs]["input_shape"][-3:] = x.shape[-3:]
            obs_meta[vis_obs]["crop_idx"] = crop_idx
            obs_dict[vis_obs] = x

        return obs_dict, obs_meta

    def unregister_hooks(self):
        if not self.is_registered:
            return
        for k, v in self.extractors.items():
            v.unregister_hooks()
        self.is_registered = False

    def register_hooks(self):
        if self.is_registered:
            return
        for k, v in self.extractors.items():
            v.register_hooks()
        self.is_registered = True

    def get_normalizer(self, obs_key):
        if self.normalizer is not None:
            return self.normalizer[obs_key]
        return None

    def initialize_extractors_and_buffers(self, **kwargs):
        extractors = {}
        buffers = {}
        obs_encoder = self.get_obs_encoder()
        for k, v in obs_encoder.obs_nets.items():
            if isinstance(v, VisualCore):
                extractors[k] = FullGrad(v, k, EncoderOnly)
                if not self.disable_buffer:
                    print(f"Obs Modality: {k}")
                    buffers[k] = BufferManager(**kwargs)
        return extractors, buffers

    def get_obs_encoder(self):
        if hasattr(self.model, "obs_nets"):
            return self.model
        elif hasattr(self.model, "nets"):
            return self.model.nets["encoder"].nets["obs"]
        else:
            raise ValueError("obs_encoder cannot be found in the model")

    def check_augmentation_strategy(self):
        assert self.aug_strategy in [
            "robosaga",
            "random_overlay",
            "erase",
            "erase_overlay",
        ], f"Invalid augmentation strategy: {self.aug_strategy}"
        assert self.aug_ratio is not None, "aug_ratio is required"
        if self.aug_strategy == "random_overlay":
            if not self.disable_buffer:
                self.disable_buffer = True
                print("SaGA Warning: Buffer is disabled for random_overlay strategy")

    def check_required_args(self, print_args=False):
        required_args = [
            "aug_strategy",
            "aug_ratio",
            "warmup_epochs",
            "saliency_erase_threshold",
            "saliency_saga_cap",
            "blend_alpha",
            "update_ratio",
            "disable_buffer",
            "output_shape",
            "backgrounds",
            "save_dir",
        ]

        for arg in required_args:
            if arg not in self.__dict__:
                raise ValueError(f"Argument {arg} is required for RoboSaGA")
        if print_args:
            print("\n=============== Saliency-guided Augmentation Parameters ===============")
            for arg in required_args:
                if isinstance(self.__dict__[arg], torch.Tensor):
                    print(f"{arg} shape: {list(self.__dict__[arg].shape)}")
                else:
                    print(f"{arg}: {self.__dict__[arg]}")
                print("\n")
