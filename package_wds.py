"""Pack a 3D Latent Traversal dataset into WebDataset tar shards.

Each shard is a standard tar archive.  One WebDataset *sample* equals one full
sequence: all T JPEG frames stored as raw bytes (no re-encode), the latent
matrix, base latent, traversal velocities, and a small JSON metadata blob.

A self-contained dataset_info.json is written alongside the shards so the
loader never needs the original metadata.pkl.

Usage (no split):
    python package_wds.py \\
        --dataset-dir ./3D_latent_traversal \\
        --output-dir  ./3D_latent_traversal_wds \\
        --shard-maxcount 512

Usage (train / val / test split):
    python package_wds.py \\
        --dataset-dir ./3D_latent_traversal \\
        --output-dir  ./3D_latent_traversal_wds \\
        --val-fraction 0.1 \\
        --test-fraction 0.1 \\
        --split-seed 42

When any fraction > 0, the script writes to subdirectories (train/, val/,
test/) each with their own shards and dataset_info.json.  Only subdirectories
that receive at least one sequence are created.
The split is a random shuffle of sequences (not objects).
"""

import argparse
import io
import json
import os
import pickle
import random
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


def _write_split(sequences, out_dir, info_base, dataset_dir, n_frames,
                 shard_pattern, split_prefix, shard_maxcount, verbose):
    """Write one set of sequences to out_dir as WebDataset shards + dataset_info.json."""
    os.makedirs(out_dir, exist_ok=True)

    info = dict(info_base)
    info["n_sequences"] = len(sequences)
    info_path = os.path.join(out_dir, "dataset_info.json")
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Wrote {info_path}")

    pattern = shard_pattern
    if split_prefix:
        pattern = f"{split_prefix}-{pattern}"
    shard_path = os.path.join(out_dir, pattern)

    n_total   = len(sequences)
    n_written = 0
    print(f"Packing {n_total} sequences → {shard_path}")

    with wds.ShardWriter(shard_path, maxcount=shard_maxcount) as sink:
        for idx, seq in enumerate(sequences):
            sample: dict = {"__key__": f"{idx:08d}"}

            # Raw JPEG bytes — read straight from disk, no decode/re-encode
            seq_dir = os.path.join(dataset_dir, seq["frames_dir"])
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

            if verbose and n_written % 100 == 0:
                print(f"  {n_written}/{n_total}")

    return n_written


def main():
    parser = argparse.ArgumentParser(
        description="Pack a 3D Latent Traversal dataset into WebDataset tar shards."
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Root directory containing metadata.pkl and seqs/")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write shard tars and dataset_info.json")
    parser.add_argument("--shard-maxcount", type=int, default=1000,
                        help="Maximum number of sequences per shard (default: 512)")
    parser.add_argument("--shard-pattern", default="shard-%05d.tar",
                        help="Shard filename pattern (default: shard-%%05d.tar)")
    parser.add_argument("--split", default=None,
                        help="Optional name prefix for shard files, "
                             "e.g. 'train' produces train-shard-%%05d.tar")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Fraction of sequences held out for validation "
                             "(default: 0).  Writes to output-dir/val/ when > 0.")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Fraction of sequences held out for testing "
                             "(default: 0).  Writes to output-dir/test/ when > 0.")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="RNG seed for the split shuffle (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Print progress every 100 sequences")
    args = parser.parse_args()

    if not 0.0 <= args.val_fraction < 1.0:
        sys.exit("ERROR: --val-fraction must be in [0, 1)")
    if not 0.0 <= args.test_fraction < 1.0:
        sys.exit("ERROR: --test-fraction must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        sys.exit("ERROR: --val-fraction + --test-fraction must be < 1")

    pkl_path = os.path.join(args.dataset_dir, "metadata.pkl")
    if not os.path.isfile(pkl_path):
        sys.exit(f"ERROR: metadata.pkl not found at {pkl_path}")

    with open(pkl_path, "rb") as fh:
        meta = pickle.load(fh)

    # ── base info dict (n_sequences filled in per split) ─────────────────────
    info_base: dict = {
        "latent_names":  meta["latent_names"],
        "latent_ranges": np.array(meta["latent_ranges"]).tolist(),
        "n_frames":      meta["n_frames"],
        "image_size":    meta["image_size"],
    }
    if "config" in meta:
        info_base["config"] = meta["config"]
    elif "configs" in meta:
        info_base["configs"] = meta["configs"]

    sequences = list(meta["sequences"])
    n_frames  = meta["n_frames"]
    n_total   = len(sequences)

    # Always shuffle so shard contents are not grouped by object.
    rng = random.Random(args.split_seed)
    rng.shuffle(sequences)

    do_split = args.val_fraction > 0.0 or args.test_fraction > 0.0

    if do_split:
        shuffled = sequences

        n_test  = max(1, round(n_total * args.test_fraction)) if args.test_fraction > 0 else 0
        n_val   = max(1, round(n_total * args.val_fraction))  if args.val_fraction  > 0 else 0
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            sys.exit("ERROR: no sequences left for training after val/test split")

        train_seqs = shuffled[:n_train]
        val_seqs   = shuffled[n_train : n_train + n_val]
        test_seqs  = shuffled[n_train + n_val :]

        print(f"Split summary (seed={args.split_seed}):")
        print(f"  train: {len(train_seqs)} sequences")
        if val_seqs:
            print(f"  val:   {len(val_seqs)} sequences")
        if test_seqs:
            print(f"  test:  {len(test_seqs)} sequences")

        splits = [("train", train_seqs)]
        if val_seqs:
            splits.append(("val", val_seqs))
        if test_seqs:
            splits.append(("test", test_seqs))

        totals = []
        for name, seqs in splits:
            n = _write_split(
                seqs, os.path.join(args.output_dir, name),
                info_base, args.dataset_dir, n_frames,
                args.shard_pattern, args.split, args.shard_maxcount, args.verbose,
            )
            totals.append(f"{n} {name}")

        print(f"\nDone.  {' + '.join(totals)} sequences written to {args.output_dir}")

    else:
        # ── no split — original behaviour ─────────────────────────────────────
        os.makedirs(args.output_dir, exist_ok=True)

        info_base["n_sequences"] = n_total
        info_path = os.path.join(args.output_dir, "dataset_info.json")
        with open(info_path, "w") as fh:
            json.dump(info_base, fh, indent=2)
        print(f"Wrote {info_path}")

        n_written = _write_split(
            sequences, args.output_dir, info_base, args.dataset_dir, n_frames,
            args.shard_pattern, args.split, args.shard_maxcount, args.verbose,
        )
        print(f"\nDone.  {n_written} sequences written to {args.output_dir}")


if __name__ == "__main__":
    main()
