import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TraversalDataset(Dataset):
    """PyTorch Dataset for the 3DLT temporal latent traversal dataset.

    Each item is one full traversal sequence: a single ShapeNet object with
    one or more latent factors swept linearly across T frames while others are fixed.

    Args:
        pkl_path:  Path to the ``metadata.pkl`` produced by generate_traversals.py.
        root_dir:  Root output directory (the parent of ``seqs/`` and ``metadata.pkl``).
        transform: Optional callable applied to each PIL image before stacking.
                   Receives a PIL.Image, should return a Tensor of shape [C, H, W].
                   Defaults to a plain ToTensor conversion (values in [0, 1]).
    """

    def __init__(self, pkl_path, root_dir, transform=None):
        with open(pkl_path, 'rb') as f:
            self._meta = pickle.load(f)

        self.root_dir = Path(root_dir)
        self.sequences = self._meta['sequences']
        self.latent_names = self._meta['latent_names']
        self.latent_ranges = torch.tensor(self._meta['latent_ranges'],
                                          dtype=torch.float32)  # [7, 2]
        self.n_frames = self._meta['n_frames']
        self.image_size = self._meta['image_size']

        if transform is None:
            from torchvision.transforms.functional import to_tensor
            self._to_tensor = to_tensor
        else:
            self._to_tensor = transform

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_dir = self.root_dir / seq['frames_dir']

        # Load frames in order
        frame_tensors = []
        for t in range(self.n_frames):
            img_path = seq_dir / f'frame_{t:04d}.jpg'
            img = Image.open(img_path).convert('RGB')
            frame_tensors.append(self._to_tensor(img))

        frames = torch.stack(frame_tensors, dim=0)   # [T, C, H, W]
        latents = torch.tensor(seq['latents'], dtype=torch.float32)       # [T, 7]
        base_latent = torch.tensor(seq['base_latent'], dtype=torch.float32)  # [7]

        directions = torch.tensor(seq['traversal_directions'], dtype=torch.int8)  # [7]

        return {
            'frames': frames,                              # FloatTensor [T, 3, H, W]
            'latents': latents,                            # FloatTensor [T, 7]
            'base_latent': base_latent,                    # FloatTensor [7]
            'synset_id': seq['synset_id'],                 # str
            'obj_id': seq['obj_id'],                       # str
            'traversal_factors': seq['traversal_factors'],     # list[int]
            'traversal_directions': directions,            # Int8Tensor [7]: +1/-1/0
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def factor_name(self, idx):
        """Return the human-readable name for latent factor index idx."""
        return self.latent_names[idx]

    def filter_by_factor(self, factor):
        """Return a view containing only sequences where the given factor varies.

        Args:
            factor: int (0-6) or str name (e.g. 'rot_x').
        Returns:
            A _SubsetDataset whose sequences all include the requested factor.
            In single-factor mode this means exactly one varying factor;
            in multi-factor mode a sequence may vary additional factors too.
        """
        if isinstance(factor, str):
            factor = self.latent_names.index(factor)
        subset = _SubsetDataset(self, [
            i for i, s in enumerate(self.sequences)
            if factor in s['traversal_factors']
        ])
        return subset

    def filter_by_synset(self, synset_id):
        """Return a view containing only sequences for the given synset category."""
        subset = _SubsetDataset(self, [
            i for i, s in enumerate(self.sequences)
            if s['synset_id'] == synset_id
        ])
        return subset


class _SubsetDataset(Dataset):
    """Lightweight index-based view into a TraversalDataset."""

    def __init__(self, base_dataset, indices):
        self._base = base_dataset
        self._indices = indices

        # Expose the same metadata attributes for convenience
        self.latent_names = base_dataset.latent_names
        self.latent_ranges = base_dataset.latent_ranges
        self.n_frames = base_dataset.n_frames
        self.image_size = base_dataset.image_size
        self.sequences = [base_dataset.sequences[i] for i in indices]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._base[self._indices[idx]]
