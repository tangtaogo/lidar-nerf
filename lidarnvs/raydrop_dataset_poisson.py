import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RaydropDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        self.split = split
        if not self.data_dir.is_dir():
            raise ValueError(f"Directory {self.data_dir} does not exist.")
        if self.split not in ["train", "test"]:
            raise ValueError(f"Split {self.split} not supported.")

        pkl_path = self.data_dir / f"{self.split}_data.pkl"
        if not pkl_path.is_file():
            raise ValueError(f"File {pkl_path} does not exist.")
        with open(pkl_path, "rb") as f:
            self.raydrop_data = pickle.load(f)

    def __len__(self):
        return len(self.raydrop_data)

    def __getitem__(self, idx):
        return self.raydrop_data[idx]

    @staticmethod
    def collate_fn(batch):
        """
        RaydropDataset is a dict-style dataset, where __getitem__(i) returns
        a dictionary of tensors. Essentially, A dataloader will do:
        ```python
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        ```

        Args:
            batch: list of dicts.

        Return:
            images: (N, C, H, W) tensor, float32.
            masks : (N, H, W) tensor, float32. 1 means valid ray.
        """
        # First, call the default colllate_fn.
        batch = torch.utils.data.default_collate(batch)

        # (N, H, W, C)
        images = torch.cat(
            [
                batch["hit_masks"][..., None],
                batch["hit_depths"][..., None],
                batch["hit_normals"],
                batch["hit_incidences"][..., None],
                batch["intensities"][..., None],
                batch["rays_d"],
            ],
            dim=3,
        )
        # (N, C, H, W)
        images = images.permute(0, 3, 1, 2)

        # (N, H, W)
        masks = batch["raydrop_masks"]

        return images, masks
