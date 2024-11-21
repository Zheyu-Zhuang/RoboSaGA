import torch

import robomimic.utils.obs_utils as ObsUtils


class BufferManager:
    """
    A class to manage saliency buffers for data augmentation.

    Attributes:
        disable_buffer (bool): Flag to disable the buffer.
        keeper (dict): Dictionary to store saliency maps.
        watcher (dict): Dictionary to track the progress of updates.
        depth (int): Depth of the buffer.
        shape (tuple): Shape of the buffer.
    """

    def __init__(self, device="cuda", **kwargs):
        """
        Initializes the BufferManager with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.depth = kwargs.get("buffer_depth", None)
        self.shape = kwargs.get("buffer_shape", None)
        self.output_shape = kwargs.get("output_shape", None)
        assert self.depth is not None, "Buffer depth must be provided"
        self.update_counter = torch.zeros(self.depth, device=device)
        H, W = self.shape
        self.buffer = torch.ones(self.depth, 1, H, W, device=device).mul_(255).to(torch.uint8)

    def set(self, smaps, buffer_ids, crop_inds=None):
        """
        Sets the saliency maps in the buffer.

        Args:
            smaps (torch.Tensor): Saliency maps.
            buffer_ids (torch.Tensor): Buffer IDs.
            obs_key (str): Observation key.
            crop_inds (torch.Tensor, optional): Crop indices.

        Raises:
            AssertionError: If saliency maps and buffer IDs size mismatch or saliency maps are not in [0, 1] range.
        """
        assert smaps.shape[0] == buffer_ids.shape[0], "Saliency and IDs size mismatch"
        assert (
            smaps.min() >= 0 and smaps.max() <= 1
        ), "Saliency maps not in [0, 1] range"

        # Remove duplicates to avoid repetitive saliency update
        unique_ids = self._first_occurrence_indices(buffer_ids)
        buffer_ids = buffer_ids[unique_ids]
        crop_inds = crop_inds[unique_ids] if crop_inds is not None else None

        smaps = smaps[unique_ids]
        smaps = (smaps * 255).to(torch.uint8)
        
        if crop_inds is not None:
            padded_smaps = torch.zeros(
                buffer_ids.shape[0], 1, self.shape[0], self.shape[1], device=smaps.device
            )
            for i in range(buffer_ids.shape[0]):
                h_0, w_0 = crop_inds[i, 0, 0], crop_inds[i, 0, 1]
                h_1, w_1 = h_0 + smaps.shape[-2], w_0 + smaps.shape[-1]
                padded_smaps[i, :, h_0:h_1, w_0:w_1] = smaps[i]
            smaps = padded_smaps

        self.buffer[buffer_ids] = smaps.to(torch.uint8)
        self.update_counter[buffer_ids] += 1

    def get(self, buffer_ids, crop_inds=None):
        """
        Retrieves the saliency maps from the buffer.

        Args:
            obs_key (str): Observation key.
            buffer_ids (torch.Tensor): Buffer IDs.
            out_shape (tuple): Output shape.
            crop_inds (torch.Tensor, optional): Crop indices.

        Raises:
            AssertionError: If crop indices and buffer IDs size mismatch.

        Returns:
            torch.Tensor: The retrieved saliency maps.
        """
        if crop_inds is not None:
            assert (
                crop_inds.shape[0] == buffer_ids.shape[0]
            ), "Crop indices and IDs size mismatch"

        # Retrieve saliency map from buffer and convert to [0, 1] range
        smaps = self.buffer[buffer_ids] / 255.0

        if crop_inds is not None:
            if self.output_shape is None:
                return smaps
            h_out, w_out = self.output_shape
            smaps = ObsUtils.crop_image_from_indices(
                smaps, crop_inds, h_out, w_out
            ).squeeze(1)
        else:
            t_h, t_w = self.output_shape
            smaps = ObsUtils.center_crop(smaps, t_h, t_w).squeeze(1)
        return smaps

    @staticmethod
    def _first_occurrence_indices(buffer_ids):
        """
        Finds the first occurrence indices of buffer IDs.

        Args:
            buffer_ids (torch.Tensor): Buffer IDs.

        Returns:
            torch.Tensor: Indices of the first occurrences.
        """
        id_dict = {}
        buffer_ids = buffer_ids.tolist()
        for i, id_ in enumerate(buffer_ids):
            if id_ not in id_dict:
                id_dict[id_] = i
        return torch.tensor([id_dict[id_] for id_ in buffer_ids])
