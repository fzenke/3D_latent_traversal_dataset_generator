"""Pack a 3D Latent Traversal dataset into WebDataset tar shards.

Each shard is a standard tar archive.  One WebDataset *sample* equals one full
sequence: all T JPEG frames stored as raw bytes (no re-encode), the latent
matrix, base latent, traversal velocities, and a small JSON metadata blob.

A self-contained dataset_info.json is written alongside the shards so the
loader never needs the original metadata.pkl.

Usage:
    python package_wds.py \\
        --dataset-dir ./3D_latent_traversal \\
        --output-dir  ./3D_latent_traversal_wds \\
        --shard-maxcount 512
"""

import argparse
import io
import json
import os
import pickle
import sys

import numpy as np

try:
    import webdataset as wds
except ImportError:
    sys.exit("webdataset not installed — run:  pip install webdataset")


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Pack a 3D Latent Traversal dataset into WebDataset tar shards."
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Root directory containing metadata.pkl and seqs/")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write shard tars and dataset_info.json")
    parser.add_argument("--shard-maxcount", type=int, default=512,
                        help="Maximum number of sequences per shard (default: 512)")
    parser.add_argument("--shard-pattern", default="shard-%05d.tar",
                        help="Shard filename pattern (default: shard-%%05d.tar)")
    parser.add_argument("--split", default=None,
                        help="Optional name prefix for shard files, "
                             "e.g. 'train' produces train-shard-%%05d.tar")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Print progress every 100 sequences")
    args = parser.parse_args()

    pkl_path = os.path.join(args.dataset_dir, "metadata.pkl")
    if not os.path.isfile(pkl_path):
        sys.exit(f"ERROR: metadata.pkl not found at {pkl_path}")

    with open(pkl_path, "rb") as fh:
        meta = pickle.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── dataset_info.json — global metadata, self-contained ──────────────────
    info: dict = {
        "latent_names":  meta["latent_names"],
        "latent_ranges": np.array(meta["latent_ranges"]).tolist(),
        "n_frames":      meta["n_frames"],
        "image_size":    meta["image_size"],
        "n_sequences":   len(meta["sequences"]),
    }
    if "config" in meta:
        info["config"] = meta["config"]
    elif "configs" in meta:
        info["configs"] = meta["configs"]

    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Wrote {info_path}")

    # ── shard writer ─────────────────────────────────────────────────────────
    pattern = args.shard_pattern
    if args.split:
        pattern = f"{args.split}-{pattern}"
    shard_path = os.path.join(args.output_dir, pattern)

    sequences = meta["sequences"]
    n_frames  = meta["n_frames"]
    n_total   = len(sequences)

    print(f"Packing {n_total} sequences → {shard_path}")
    print(f"  shard-maxcount: {args.shard_maxcount}")

    n_written = 0
    with wds.ShardWriter(shard_path, maxcount=args.shard_maxcount) as sink:
        for idx, seq in enumerate(sequences):
            sample: dict = {"__key__": f"{idx:08d}"}

            # Raw JPEG bytes — read straight from disk, no decode/re-encode
            seq_dir = os.path.join(args.dataset_dir, seq["frames_dir"])
            for t in range(n_frames):
                frame_path = os.path.join(seq_dir, f"frame_{t:04d}.jpg")
                if not os.path.isfile(frame_path):
                    sys.exit(f"ERROR: missing frame {frame_path}")
                with open(frame_path, "rb") as fh:
                    sample[f"frame_{t:04d}.jpg"] = fh.read()

            # Numpy arrays
            sample["latents.npy"]     = _npy_bytes(seq["latents"].astype(np.float32))
            sample["base_latent.npy"] = _npy_bytes(seq["base_latent"].astype(np.float32))
            sample["velocities.npy"]  = _npy_bytes(
                np.asarray(seq["traversal_velocities"], dtype=np.float32)
            )

            # Per-sequence metadata
            sample["meta.json"] = json.dumps({
                "synset_id":         seq["synset_id"],
                "obj_id":            seq["obj_id"],
                "seq_idx":           seq["seq_idx"],
                "traversal_factors": seq["traversal_factors"],
            }).encode()

            sink.write(sample)
            n_written += 1

            if args.verbose and n_written % 100 == 0:
                print(f"  {n_written}/{n_total}")

    print(f"\nDone.  {n_written} sequences written to {args.output_dir}")


if __name__ == "__main__":
    main()
