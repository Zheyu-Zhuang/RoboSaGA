import os
import random
from PIL import Image

import torch
import torch.nn.functional as F
import kornia as K
from torchvision import transforms
from tqdm import tqdm


class BackgroundRandomizer:
    """
    A class to randomize backgrounds for data augmentation.

    Attributes:
        im_size (tuple): The size of the images (height, width).
        backgrounds (torch.Tensor): Preloaded background images.
    """
    def __init__(self, **kwargs):
        assert "output_shape" in kwargs, "output_shape must be provided"
        assert "background_path" in kwargs, "background_path must be provided"
        self.im_size = kwargs["output_shape"]
        self.backgrounds = self.preload_all(kwargs["background_path"])

    def __call__(self, num_samples, normalizer=None):
        """
        Generates random backgrounds and applies normalization if provided.

        Args:
            num_samples (int): The number of samples to generate.
            normalizer (callable, optional): A function to normalize the images.

        Returns:
            torch.Tensor: The generated background images.
        """
        rand_bg_idx = random.sample(range(self.backgrounds.shape[0]), num_samples)
        bg = self.random_transform(self.backgrounds[rand_bg_idx])
        return normalizer(bg) if normalizer is not None else bg

    def __str__(self):
        return f"BackgroundRandomizer: {self.backgrounds.shape[0]} backgrounds, size {self.im_size}"

    def random_transform(self, bg):
        """
        Applies random transformations to the background images.
            1. Random rotation from 0 to 360 degrees.
            2. Random brightness from -0.1 to 0.1.
        Args:
            bg (torch.Tensor): The background images to transform.

        Returns:
            torch.Tensor: The transformed background images.
        """
        if bg.size(2) != self.im_size[0] or bg.size(3) != self.im_size[1]:
            bg = F.interpolate(bg, size=self.im_size, mode="bilinear")

        angles = torch.rand(bg.size(0)) * 360  # Random angle from 0 to 360 degrees
        rotated_images = K.geometry.transform.rotate(
            bg, angles.to(bg.device), mode="bilinear", padding_mode="border"
        )

        brightness_factors = (
            torch.rand(bg.size(0)) * 0.2
        ) - 0.1  # Random brightness from -0.1 to 0.1
        bg = K.enhance.adjust_brightness(rotated_images, brightness_factors)

        return bg

    def preload_all(self, background_path):
        """
        Preloads all background images from the specified directory.

        Args:
            background_path (str): The directory containing background images.

        Returns:
            torch.Tensor: The preloaded background images.
        """
        print("Preloading Background Images...")
        all_f_names = os.listdir(background_path)
        all_im_names = [
            f_name for f_name in all_f_names if f_name.endswith((".jpg", ".png"))
        ]

        n_images = len(all_im_names)
        backgrounds = torch.zeros(
            (n_images, 3, self.im_size[0], self.im_size[1]), dtype=torch.float32
        )

        for i, im_name in enumerate(tqdm(all_im_names, desc="Loading images")):
            im_path = os.path.join(background_path, im_name)
            try:
                im = Image.open(im_path).convert("RGB")
                if im.size != self.im_size:
                    im = im.resize(self.im_size)
                im_tensor = transforms.ToTensor()(im)
                backgrounds[i] = im_tensor
            except Exception as e:
                print(f"Error loading image {im_path}: {e}")

        return backgrounds.to("cuda")
