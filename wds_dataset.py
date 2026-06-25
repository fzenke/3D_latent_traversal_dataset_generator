"""PyTorch DataLoader for WebDataset-packaged 3D Latent Traversal shards.

Mirrors the TraversalDataset dict API so training code works unchanged:

    frames               FloatTensor  [T, 3, H, W]   values in [0, 1]
    latents              FloatTensor  [T, 10]
    base_latent          FloatTensor  [10]
    traversal_velocities FloatTensor  [10]            0 = frozen, ±v = sweep speed
    synset_id            str
    obj_id               str
    traversal_factors    list[int]                    indices of varying factors

Shards are produced by package_wds.py.

Usage:
    from wds_dataset import TraversalWebDataset, make_loader

    ds     = TraversalWebDataset("./shards/shard-{00000..00020}.tar")
    loader = make_loader(ds, batch_size=16, num_workers=4)

    for batch in loader:
        frames = batch["frames"]   # [B, T, 3, H, W]
        ...
"""

import io
import json
import os

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision.transforms.functional import to_tensor


# ── per-sample decode ─────────────────────────────────────────────────────────

def _decode_sample(raw: dict) -> dict:
    """Decode raw byte fields from a WebDataset sample."""
    out: dict = {"__key__": raw["__key__"]}
    for k, v in raw.items():
        if k == "__key__":
            continue
        if k.endswith(".jpg"):
            out[k] = to_tensor(Image.open(io.BytesIO(v)).convert("RGB"))  # [3, H, W]
        elif k.endswith(".npy"):
            out[k] = np.load(io.BytesIO(v))
        elif k.endswith(".json"):
            out[k] = json.loads(v if isinstance(v, str) else v.decode())
        else:
            out[k] = v
    return out


def _build_sequence(decoded: dict) -> dict:
    """Stack frame tensors and assemble the final per-sequence dict."""
    frame_keys = sorted(k for k in decoded if k.startswith("frame_") and k.endswith(".jpg"))
    frames = torch.stack([decoded[k] for k in frame_keys])  # [T, 3, H, W]
    meta   = decoded["meta.json"]
    return {
        "frames":               frames,
        "latents":              torch.from_numpy(decoded["latents.npy"]),
        "base_latent":          torch.from_numpy(decoded["base_latent.npy"]),
        "traversal_velocities": torch.from_numpy(decoded["velocities.npy"]),
        "synset_id":            meta["synset_id"],
        "obj_id":               meta["obj_id"],
        "traversal_factors":    meta["traversal_factors"],
    }


# ── batching ──────────────────────────────────────────────────────────────────

def collate_sequences(samples: list) -> dict:
    """Collate a list of sequence dicts into a padded batch.

    Tensors are stacked along dim 0.  String fields and traversal_factors
    (variable-length lists) are kept as Python lists.
    """
    return {
        "frames":               torch.stack([s["frames"] for s in samples]),
        "latents":              torch.stack([s["latents"] for s in samples]),
        "base_latent":          torch.stack([s["base_latent"] for s in samples]),
        "traversal_velocities": torch.stack([s["traversal_velocities"] for s in samples]),
        "synset_id":            [s["synset_id"] for s in samples],
        "obj_id":               [s["obj_id"] for s in samples],
        "traversal_factors":    [s["traversal_factors"] for s in samples],
    }


# ── dataset info ──────────────────────────────────────────────────────────────

def load_dataset_info(shard_dir: str) -> dict:
    """Load global metadata from dataset_info.json written by package_wds.py."""
    info_path = os.path.join(shard_dir, "dataset_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"dataset_info.json not found in {shard_dir}")
    with open(info_path) as fh:
        info = json.load(fh)
    info["latent_ranges"] = torch.tensor(info["latent_ranges"], dtype=torch.float32)
    return info


# ── main class ────────────────────────────────────────────────────────────────

class TraversalWebDataset:
    """WebDataset pipeline builder for 3D Latent Traversal tar shards.

    This is not a PyTorch Dataset subclass (WebDataset pipelines are iterable,
    not indexed).  Call .dataset() to get the pipeline, then wrap it with
    make_loader() or wds.WebLoader directly.

    Args:
        urls:          Glob pattern, brace-expanded WebDataset URL string, or
                       list of shard paths.
                       Examples:
                         "./shards/shard-{00000..00020}.tar"  (brace expansion)
                         "./shards/shard-*.tar"               (glob)
                         ["./shards/shard-00000.tar", ...]    (list)
        info_dir:      Directory containing dataset_info.json.  Inferred as
                       the parent directory of the URL pattern when omitted.
        transform:     Optional callable applied to each frame tensor
                       [3, H, W] → [3, H, W] before stacking.
        shuffle:       In-epoch sample shuffle buffer size (0 = no shuffle).
        shardshuffle:  If True, shard order is shuffled between epochs.
    """

    def __init__(
        self,
        urls,
        info_dir: str = None,
        transform=None,
        shuffle: int = 1000,
        shardshuffle: bool = True,
    ):
        self._urls         = urls
        self._transform    = transform
        self._shuffle      = shuffle
        self._shardshuffle = shardshuffle

        # Infer info_dir from URL pattern
        if info_dir is None:
            if isinstance(urls, list):
                info_dir = os.path.dirname(urls[0]) if urls else "."
            else:
                info_dir = os.path.dirname(urls) or "."

        try:
            info = load_dataset_info(info_dir)
        except FileNotFoundError:
            info = {}

        self.latent_names  = info.get("latent_names", [])
        self.latent_ranges = info.get("latent_ranges", None)   # FloatTensor [N, 2]
        self.n_frames      = info.get("n_frames", None)
        self.image_size    = info.get("image_size", None)
        self.n_sequences   = info.get("n_sequences", None)

    # ── pipeline builders ─────────────────────────────────────────────────────

    def _base_pipeline(self):
        transform = self._transform

        def _process(raw):
            sample = _build_sequence(_decode_sample(raw))
            if transform is not None:
                sample["frames"] = torch.stack(
                    [transform(sample["frames"][t])
                     for t in range(sample["frames"].shape[0])]
                )
            return sample

        pipeline = wds.WebDataset(self._urls, shardshuffle=self._shardshuffle)
        if self._shuffle > 0:
            pipeline = pipeline.shuffle(self._shuffle)
        return pipeline.map(_process)

    def dataset(self) -> wds.WebDataset:
        """Return the full pipeline (all sequences)."""
        return self._base_pipeline()

    def filter_by_factor(self, factor) -> wds.WebDataset:
        """Return a pipeline restricted to sequences where factor varies.

        Args:
            factor: int index (0–9) or str name, e.g. 'rot_x'.
        """
        if isinstance(factor, str):
            factor = self.latent_names.index(factor)
        return self._base_pipeline().select(
            lambda s, f=factor: f in s["traversal_factors"]
        )

    def filter_by_synset(self, synset_id: str) -> wds.WebDataset:
        """Return a pipeline restricted to a single ShapeNet synset."""
        return self._base_pipeline().select(
            lambda s, sid=synset_id: s["synset_id"] == sid
        )

    def factor_name(self, idx: int) -> str:
        return self.latent_names[idx]


# ── DataLoader helper ─────────────────────────────────────────────────────────

def make_loader(
    source,
    batch_size: int = 16,
    num_workers: int = 4,
    **twd_kwargs,
) -> wds.WebLoader:
    """Build a DataLoader from a TraversalWebDataset, a pipeline, or shard URLs.

    Args:
        source:       TraversalWebDataset instance, a wds pipeline, or a URL
                      string / list of paths.
        batch_size:   Sequences per batch.
        num_workers:  Worker processes for parallel shard reading.
        **twd_kwargs: Forwarded to TraversalWebDataset when source is a URL.
    """
    if isinstance(source, TraversalWebDataset):
        pipeline = source.dataset()
    elif isinstance(source, (str, list)):
        pipeline = TraversalWebDataset(source, **twd_kwargs).dataset()
    else:
        pipeline = source   # already a wds pipeline

    return wds.WebLoader(
        pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_sequences,
    )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the WebDataset traversal loader."
    )
    parser.add_argument("--shards", required=True,
                        help="Glob or brace URL, e.g. './shards/shard-*.tar'")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-batches", type=int, default=5,
                        help="Number of batches to load for this test")
    args = parser.parse_args()

    twd = TraversalWebDataset(args.shards, shuffle=200)
    print(f"latent_names : {twd.latent_names}")
    print(f"n_frames     : {twd.n_frames}")
    print(f"image_size   : {twd.image_size}")
    print(f"n_sequences  : {twd.n_sequences}")

    loader = make_loader(twd, batch_size=args.batch_size, num_workers=args.num_workers)

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        print(f"\nbatch {i}")
        print(f"  frames               {tuple(batch['frames'].shape)}"
              f"  dtype={batch['frames'].dtype}")
        print(f"  latents              {tuple(batch['latents'].shape)}")
        print(f"  base_latent          {tuple(batch['base_latent'].shape)}")
        print(f"  traversal_velocities {tuple(batch['traversal_velocities'].shape)}")
        print(f"  synset_id            {batch['synset_id']}")
        print(f"  traversal_factors    {batch['traversal_factors']}")
