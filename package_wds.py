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

--split-by controls what the held-out splits actually measure:

    sequence  (default)  Sequences are shuffled at random, so every object
                         appears in every split.  Held-out metrics measure
                         generalisation to unseen *latent configurations*.

    object               Whole objects are held out, stratified per synset, so
                         no object appears in more than one split.  Held-out
                         metrics measure generalisation to *unseen objects*.

Both modes record their settings in dataset_info.json, so a packaged dataset
says which kind of split it carries.
"""

import argparse
import io
import json
import os
import pickle
import random
import sys
from collections import defaultdict

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


def split_sequences(sequences, val_fraction, test_fraction, seed=0,
                    split_by="sequence", warn=print):
    """Partition sequences into train / val / test.

    split_by='sequence'
        Random shuffle of sequences.  Every object appears in every split, so
        held-out metrics measure generalisation to unseen *latent
        configurations* of objects the model has already seen.

    split_by='object'
        Whole objects are held out: all sequences of a given (synset_id,
        obj_id) land in the same split, so held-out metrics measure
        generalisation to *unseen objects*.  The partition is stratified per
        synset, so every category is represented in every split as long as it
        has enough objects.  Split sizes are then only approximately equal to
        the requested fractions, because objects are indivisible.

    Returns {"train": [...], "val": [...], "test": [...]}; empty splits are
    omitted.  Sequences within each split are shuffled so shard contents are
    never grouped by object.
    """
    rng = random.Random(seed)

    if split_by == "sequence":
        shuffled = list(sequences)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = max(1, round(n_total * test_fraction)) if test_fraction > 0 else 0
        n_val  = max(1, round(n_total * val_fraction))  if val_fraction  > 0 else 0
        if n_total - n_val - n_test <= 0:
            sys.exit("ERROR: no sequences left for training after val/test split")
        n_train = n_total - n_val - n_test
        parts = {
            "train": shuffled[:n_train],
            "val":   shuffled[n_train:n_train + n_val],
            "test":  shuffled[n_train + n_val:],
        }
        return {k: v for k, v in parts.items() if v}

    if split_by != "object":
        sys.exit(f"ERROR: unknown --split-by value: {split_by}")

    # ── object-level split, stratified per synset ────────────────────────────
    by_object = defaultdict(list)
    for seq in sequences:
        by_object[(seq["synset_id"], seq["obj_id"])].append(seq)

    by_synset = defaultdict(list)
    for synset_id, obj_id in by_object:
        by_synset[synset_id].append(obj_id)

    parts = {"train": [], "val": [], "test": []}
    for synset_id in sorted(by_synset):
        objs = sorted(by_synset[synset_id])
        rng.shuffle(objs)
        n_obj = len(objs)

        n_test = round(n_obj * test_fraction) if test_fraction > 0 else 0
        n_val  = round(n_obj * val_fraction)  if val_fraction  > 0 else 0
        # Give each requested split at least one object, if the synset can
        # spare it — rounding alone would drop small categories entirely.
        if test_fraction > 0 and n_test == 0 and n_obj - n_val > 1:
            n_test = 1
        if val_fraction > 0 and n_val == 0 and n_obj - n_test > 1:
            n_val = 1
        # Training must not be starved; shave from test first, then val.
        while n_obj - n_val - n_test < 1 and n_test > 0:
            n_test -= 1
        while n_obj - n_val - n_test < 1 and n_val > 0:
            n_val -= 1

        n_train = n_obj - n_val - n_test
        if test_fraction > 0 and n_test == 0:
            warn(f"  WARNING: synset {synset_id} has only {n_obj} object(s) — "
                 f"absent from the test split")
        if val_fraction > 0 and n_val == 0:
            warn(f"  WARNING: synset {synset_id} has only {n_obj} object(s) — "
                 f"absent from the val split")

        for name, chunk in (("train", objs[:n_train]),
                            ("val",   objs[n_train:n_train + n_val]),
                            ("test",  objs[n_train + n_val:])):
            for obj_id in chunk:
                parts[name].extend(by_object[(synset_id, obj_id)])

    if not parts["train"]:
        sys.exit("ERROR: no sequences left for training after val/test split")

    # Shuffle within each split so shards are not grouped by object.
    for seqs in parts.values():
        rng.shuffle(seqs)
    return {k: v for k, v in parts.items() if v}


def n_objects(sequences):
    return len({(s["synset_id"], s["obj_id"]) for s in sequences})


def objects_by_synset(sequences):
    """{synset_id: [obj_id, ...]} for the objects present in `sequences`.

    Written into each split's dataset_info.json so the membership of an
    object-level split is a recorded fact rather than something you have to
    recover by scanning tars.
    """
    out = defaultdict(set)
    for seq in sequences:
        out[seq["synset_id"]].add(seq["obj_id"])
    return {sid: sorted(objs) for sid, objs in sorted(out.items())}


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
    parser.add_argument("--split-by", choices=("sequence", "object"),
                        default="sequence",
                        help="Unit held out by --val-fraction / --test-fraction. "
                             "'sequence' (default) splits sequences at random, so "
                             "every object appears in every split and held-out "
                             "metrics measure generalisation to unseen latent "
                             "configurations.  'object' holds out whole objects "
                             "(stratified per synset), so held-out metrics measure "
                             "generalisation to unseen objects.")
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

    sequences = list(meta["sequences"])
    n_frames  = meta["n_frames"]
    n_total   = len(sequences)

    # ── base info dict (n_sequences filled in per split) ─────────────────────
    info_base: dict = {
        "latent_names":       meta["latent_names"],
        "latent_ranges":      np.array(meta["latent_ranges"]).tolist(),
        "n_frames":           meta["n_frames"],
        "image_size":         meta["image_size"],
        "unique_synset_ids":  sorted(set(s["synset_id"] for s in sequences)),
        # Recorded so a released dataset is self-describing about what its
        # held-out splits actually measure.
        "split_by":           args.split_by,
        "split_seed":         args.split_seed,
        "val_fraction":       args.val_fraction,
        "test_fraction":      args.test_fraction,
    }
    if "config" in meta:
        info_base["config"] = meta["config"]
    elif "configs" in meta:
        info_base["configs"] = meta["configs"]

    do_split = args.val_fraction > 0.0 or args.test_fraction > 0.0

    if do_split:
        print(f"Splitting by {args.split_by} (seed={args.split_seed}):")
        parts = split_sequences(sequences, args.val_fraction,
                                args.test_fraction, args.split_seed,
                                args.split_by)

        for name in ("train", "val", "test"):
            if name in parts:
                print(f"  {name + ':':<7s} {len(parts[name]):>7d} sequences  "
                      f"{n_objects(parts[name]):>5d} objects")
        if args.split_by == "object":
            print("  (objects are disjoint across splits — held-out metrics "
                  "measure generalisation to unseen objects)")
            for name in ("val", "test"):
                if name not in parts:
                    continue
                print(f"\n  held-out objects in '{name}':")
                for synset_id, obj_ids in objects_by_synset(parts[name]).items():
                    print(f"    {synset_id}  {' '.join(obj_ids)}")
            print("\n  (also written to each split's dataset_info.json "
                  "under 'objects')")

        splits = [(name, parts[name]) for name in ("train", "val", "test")
                  if name in parts]

        totals = []
        for name, seqs in splits:
            split_info = dict(info_base, n_objects=n_objects(seqs),
                              objects=objects_by_synset(seqs))
            n = _write_split(
                seqs, os.path.join(args.output_dir, name),
                split_info, args.dataset_dir, n_frames,
                args.shard_pattern, args.split, args.shard_maxcount, args.verbose,
            )
            totals.append(f"{n} {name}")

        print(f"\nDone.  {' + '.join(totals)} sequences written to {args.output_dir}")

    else:
        # ── no split — original behaviour ─────────────────────────────────────
        os.makedirs(args.output_dir, exist_ok=True)

        # Shuffle so shard contents are not grouped by object.
        random.Random(args.split_seed).shuffle(sequences)

        info_base["n_sequences"] = n_total
        info_base["n_objects"]   = n_objects(sequences)
        info_base["objects"]     = objects_by_synset(sequences)
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
