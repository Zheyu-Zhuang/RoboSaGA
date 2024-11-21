import os

import cv2
import numpy as np
import torch


def unnormalize_image(x, normalizer):
    input_shape = x.shape
    if normalizer is None:
        return x
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    x = normalizer.unnormalize(x)
    return x.view(input_shape)

def batch_to_ims(torch_ims):
    if torch_ims is None:
        return None
    is_gray = len(torch_ims.shape) == 3
    if is_gray:
        torch_ims = torch_ims.unsqueeze(1)
    np_ims = torch_ims.permute(0, 2, 3, 1).detach().cpu().numpy()
    np_ims = (np.clip(np_ims, 0, 1) * 255).astype(np.uint8)
    cv2_ims = []
    for i in range(np_ims.shape[0]):
        im_ = np_ims[i]
        if is_gray:
            im_ = cv2.applyColorMap(im_, cv2.COLORMAP_JET)
        else:
            im_ = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
        cv2_ims.append(im_)
    im_pad = np.ones((cv2_ims[0].shape[0], 4, 3), dtype=np.uint8)
    for i in range(len(cv2_ims) - 1):
        cv2_ims.insert(2 * i + 1, im_pad)
    return cv2.hconcat(cv2_ims)

def normalize_smaps(smaps):
    if len(smaps.shape) == 4:
        smaps = smaps.squeeze(1)
    s_min = torch.amin(smaps, dim=(1, 2), keepdim=True)
    s_max = torch.amax(smaps, dim=(1, 2), keepdim=True)
    smaps_norm = (smaps - s_min) / (s_max - s_min + 1e-6)
    return torch.clamp(smaps_norm, 0, 1)


def vstack_images(images, padding=2):
    ims = [pad_image(im, padding) for im in images if im is not None]
    if not ims:
        return None
    if len(ims) == 1:
        return ims[0]
    return cv2.vconcat(ims)


def pad_image(im, pad_size):
    if im is None or pad_size == 0:
        return im
    return cv2.copyMakeBorder(
        im, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
def get_title_banner(title, im_width, font_scale=0.8, font_thickness=1):
    banner = np.zeros((50, im_width, 3), dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
    return banner.astype(np.uint8)

def put_banner_on_image(im, title):
    if im is None:
        return None
    banner = get_title_banner(title, im.shape[1])
    return cv2.vconcat([banner, im])
