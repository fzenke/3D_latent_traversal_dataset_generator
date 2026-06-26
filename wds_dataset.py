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

import glob as _glob
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


def _filter_shards_by_split(urls, split: str = None):
    """Filter shards according to the train/val/test split convention."""
    if split is None:
        return urls

    if isinstance(urls, str):
        urls = [urls]

    def is_train_shard(path):
        return os.path.basename(path).startswith("train_shard_")
    
    def is_val_shard(path):
        return os.path.basename(path).startswith("val_shard_")
    
    def is_test_shard(path):
        return os.path.basename(path).startswith("test_shard_")

    if split == "train":
        filtered = [url for url in urls if is_train_shard(url)]
    elif split == "val":
        filtered = [url for url in urls if is_val_shard(url)]
    else:
        filtered = [url for url in urls if is_test_shard(url)]

    if not filtered:
        raise FileNotFoundError(f"No shard files matched split='{split}'")
    return filtered


# ── picklable callables for use in worker processes ───────────────────────────

class _SampleProcessor:
    """Decode + optional per-frame transform.  Picklable (unlike a closure)."""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, raw: dict) -> dict:
        sample = _build_sequence(_decode_sample(raw))
        if self.transform is not None:
            sample["frames"] = torch.stack(
                [self.transform(sample["frames"][t])
                 for t in range(sample["frames"].shape[0])]
            )
        return sample


class _FactorFilter:
    def __init__(self, factor_idx: int):
        self._idx = factor_idx

    def __call__(self, sample: dict) -> bool:
        return self._idx in sample["traversal_factors"]


class _SynsetFilter:
    def __init__(self, synset_id: str):
        self._sid = synset_id

    def __call__(self, sample: dict) -> bool:
        return sample["synset_id"] == self._sid


# ── batching ──────────────────────────────────────────────────────────────────

def collate_sequences(samples: list) -> dict:
    """Collate a list of sequence dicts into a batch.

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
    not indexed).  Call .dataset() or make_loader() to get something you can
    iterate.

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
        split:         Optional shard split selector.
        shuffle:       In-epoch sample shuffle buffer size (0 = no shuffle).
        shardshuffle:  If True, shard order is shuffled between epochs.
    """

    def __init__(
        self,
        urls,
        info_dir: str = None,
        transform=None,
        split: str = None,
        shuffle: int = 1000,
        shardshuffle: int = 10,
    ):
        # Expand glob patterns to a concrete list so wds.WebDataset never
        # receives a wildcard string (it would try to open "*.tar" literally).
        if isinstance(urls, str) and "*" in urls:
            expanded = sorted(_glob.glob(urls))
            if not expanded:
                raise FileNotFoundError(f"No shard files found matching: {urls}")
            self._urls = expanded
        else:
            self._urls = urls

        self.split = split
        self._urls = _filter_shards_by_split(self._urls, split=split)

        self._transform    = transform
        self._shuffle      = shuffle
        # webdataset expects shardshuffle as a non-negative integer (buffer
        # size), not a bool.  Convert True → 10 shards, False/0 → 0.
        self._shardshuffle = 10 if shardshuffle is True else int(shardshuffle)

        if info_dir is None:
            if isinstance(self._urls, list):
                info_dir = os.path.dirname(self._urls[0]) if self._urls else "."
            else:
                info_dir = os.path.dirname(self._urls) or "."

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
        pipeline = wds.WebDataset(self._urls, shardshuffle=self._shardshuffle)
        if self._shuffle > 0:
            pipeline = pipeline.shuffle(self._shuffle)
        return pipeline.map(_SampleProcessor(self._transform))

    def dataset(self, batch_size: int = None) -> wds.WebDataset:
        """Return the full pipeline.

        Args:
            batch_size: If given, batching is done inside the pipeline with
                        collate_sequences.  Pass batch_size=None (default) when
                        you handle batching in WebLoader / DataLoader yourself.
        """
        pipeline = self._base_pipeline()
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

    def filter_by_factor(self, factor, batch_size: int = None) -> wds.WebDataset:
        """Return a pipeline restricted to sequences where factor varies.

        Args:
            factor:     int index (0–9) or str name, e.g. 'rot_x'.
            batch_size: Optional in-pipeline batching (see dataset()).
        """
        if isinstance(factor, str):
            factor = self.latent_names.index(factor)
        pipeline = self._base_pipeline().select(_FactorFilter(factor))
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

    def filter_by_synset(self, synset_id: str, batch_size: int = None) -> wds.WebDataset:
        """Return a pipeline restricted to a single ShapeNet synset."""
        pipeline = self._base_pipeline().select(_SynsetFilter(synset_id))
        if batch_size is not None:
            pipeline = pipeline.batched(batch_size, collation_fn=collate_sequences)
        return pipeline

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

    Batching is moved inside the pipeline (via .batched()) so each worker
    yields complete batches.  This is the recommended WebDataset pattern for
    multi-worker loading and avoids deadlocks that arise when collation is
    done outside the pipeline.

    Args:
        source:       TraversalWebDataset instance, a wds pipeline, or a URL
                      string / list of paths.
        batch_size:   Sequences per batch.
        num_workers:  Worker processes for parallel shard reading.
        **twd_kwargs: Forwarded to TraversalWebDataset when source is a URL.
    """
    if isinstance(source, TraversalWebDataset):
        pipeline = source.dataset(batch_size=batch_size)
    elif isinstance(source, (str, list)):
        pipeline = TraversalWebDataset(source, **twd_kwargs).dataset(batch_size=batch_size)
    else:
        # Assume source is already a batched wds pipeline
        pipeline = source

    # batch_size=None: batching already happened inside the pipeline
    return wds.WebLoader(pipeline, num_workers=num_workers, batch_size=None)


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob as _glob
    import sys
    import time

    def _p(*a, **kw):
        print(*a, **kw, flush=True)

    parser = argparse.ArgumentParser(
        description="Staged smoke-test for the WebDataset traversal loader.\n\n"
                    "Runs three stages in order:\n"
                    "  Stage 1 — raw pipeline, no DataLoader\n"
                    "  Stage 2 — DataLoader with num_workers=0\n"
                    "  Stage 3 — DataLoader with --num-workers N (skipped if N=0)\n"
                    "If a stage hangs, interrupt and rerun with the flags shown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shards", required=True,
                        help="Glob or brace URL, e.g. './shards/shard-*.tar'")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers for Stage 3 (default 0 = skip Stage 3)")
    parser.add_argument("--n-batches", type=int, default=5,
                        help="Batches to read in Stage 2/3 (default 5)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable sample-level shuffle buffer "
                             "(try this if Stage 1 hangs)")
    parser.add_argument("--no-shardshuffle", action="store_true",
                        help="Disable shard-order shuffling "
                             "(try this if Stage 1 hangs)")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip Stage 1 and go straight to Stage 2")
    args = parser.parse_args()

    # ── resolve shards (TraversalWebDataset expands the glob) ────────────────
    shuffle      = 0 if args.no_shuffle else 200
    shardshuffle = 0 if args.no_shardshuffle else 10
    twd = TraversalWebDataset(args.shards, shuffle=shuffle, shardshuffle=shardshuffle)

    shard_files = twd._urls if isinstance(twd._urls, list) else [twd._urls]
    _p(f"\nFound {len(shard_files)} shard file(s) matching '{args.shards}':")
    for s in shard_files[:6]:
        _p(f"  {s}")
    if len(shard_files) > 6:
        _p(f"  ... and {len(shard_files) - 6} more")
    _p(f"\nlatent_names : {twd.latent_names}")
    _p(f"n_frames     : {twd.n_frames}")
    _p(f"image_size   : {twd.image_size}")
    _p(f"n_sequences  : {twd.n_sequences}")

    # ── Stage 1: raw pipeline, no DataLoader ─────────────────────────────────
    if not args.skip_raw:
        _p("\n=== Stage 1: raw pipeline (no workers, no batching) ===")
        _p("  If this hangs, re-run with --no-shuffle --no-shardshuffle")
        pipeline = twd.dataset()   # unbatched iterable
        t0 = time.time()
        for i, sample in enumerate(pipeline):
            elapsed = time.time() - t0
            _p(f"  sample {i}  ({elapsed:.2f}s)  "
               f"frames={tuple(sample['frames'].shape)}  "
               f"synset={sample['synset_id']}  "
               f"factors={sample['traversal_factors']}")
            t0 = time.time()
            if i >= 2:
                break
        _p("Stage 1 OK.")

    # ── Stage 2: DataLoader, num_workers=0 ───────────────────────────────────
    _p(f"\n=== Stage 2: DataLoader, batch_size={args.batch_size}, num_workers=0 ===")
    _p("  If this hangs but Stage 1 passed, file I/O in DataLoader context differs")
    loader0 = make_loader(twd, batch_size=args.batch_size, num_workers=0)
    t0 = time.time()
    for i, batch in enumerate(loader0):
        elapsed = time.time() - t0
        _p(f"  batch {i}  ({elapsed:.2f}s)  "
           f"frames={tuple(batch['frames'].shape)}  "
           f"synset={batch['synset_id']}")
        t0 = time.time()
        if i >= min(args.n_batches, 2):
            break
    _p("Stage 2 OK.")

    # ── Stage 3: DataLoader with workers ─────────────────────────────────────
    if args.num_workers > 0:
        _p(f"\n=== Stage 3: DataLoader, batch_size={args.batch_size},"
           f" num_workers={args.num_workers} ===")
        _p("  If this hangs but Stage 2 passed, workers cannot access shard files")
        _p("  (check NFS mount visibility / path resolution in child processes)")
        loader = make_loader(twd, batch_size=args.batch_size, num_workers=args.num_workers)
        t0 = time.time()
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            elapsed = time.time() - t0
            _p(f"\n  batch {i}  ({elapsed:.2f}s)")
            _p(f"    frames               {tuple(batch['frames'].shape)}"
               f"  dtype={batch['frames'].dtype}")
            _p(f"    latents              {tuple(batch['latents'].shape)}")
            _p(f"    traversal_velocities {tuple(batch['traversal_velocities'].shape)}")
            _p(f"    synset_id            {batch['synset_id']}")
            _p(f"    traversal_factors    {batch['traversal_factors']}")
            t0 = time.time()
        _p("Stage 3 OK.")
    else:
        _p("\n(Stage 3 skipped — pass --num-workers N to test multiprocessing)")
