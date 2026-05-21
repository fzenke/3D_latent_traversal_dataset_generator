import argparse
import glob
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser(
    description="Merge partial metadata_*.pkl files into a single metadata.pkl."
)
parser.add_argument('--output-dir', required=True,
                    help="Shared output directory containing metadata_*.pkl files")
parser.add_argument('--pattern', default='metadata_*.pkl',
                    help="Glob pattern for partial pickle files (default: metadata_*.pkl)")
parser.add_argument('--out-name', default='metadata',
                    help="Name for the merged output pickle, without .pkl (default: metadata)")
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help="Print per-file stats and a post-merge breakdown")
args = parser.parse_args()


def vprint(*a, **kw):
    if args.verbose:
        print(*a, **kw)


partial_paths = sorted(glob.glob(os.path.join(args.output_dir, args.pattern)))
if not partial_paths:
    raise FileNotFoundError(
        f"No files matched {os.path.join(args.output_dir, args.pattern)}"
    )
print(f"Found {len(partial_paths)} partial pickle(s)")

# Load and validate all partials
header_keys = ('latent_names', 'latent_ranges', 'n_frames', 'image_size')
reference_header = None
all_sequences = []
seen = {}  # (synset_id, obj_id, seq_idx) -> source file

for path in partial_paths:
    with open(path, 'rb') as f:
        part = pickle.load(f)

    # Extract and validate header
    header = {k: part[k] for k in header_keys}
    if reference_header is None:
        reference_header = header
        vprint(f"  Reference header from {os.path.basename(path)}:")
        vprint(f"    n_frames={header['n_frames']}  image_size={header['image_size']}")
        vprint(f"    latent_names={header['latent_names']}")
    else:
        for k in header_keys:
            ref_val = reference_header[k]
            val = header[k]
            if isinstance(ref_val, np.ndarray):
                if not np.array_equal(ref_val, val):
                    raise ValueError(
                        f"Header mismatch for '{k}' in {path}: "
                        f"expected {ref_val}, got {val}"
                    )
            elif ref_val != val:
                raise ValueError(
                    f"Header mismatch for '{k}' in {path}: "
                    f"expected {ref_val}, got {val}"
                )

    n_seqs = len(part['sequences'])
    n_objs = len({(s['synset_id'], s['obj_id']) for s in part['sequences']})
    synsets = sorted({s['synset_id'] for s in part['sequences']})
    n_dup = 0

    for seq in part['sequences']:
        key = (seq['synset_id'], seq['obj_id'], seq['seq_idx'])
        if key in seen:
            print(f"  WARNING: duplicate sequence {key} in {os.path.basename(path)} "
                  f"(first seen in {os.path.basename(seen[key])})")
            n_dup += 1
        else:
            seen[key] = path
            all_sequences.append(seq)

    vprint(f"  {os.path.basename(path):30s}  "
           f"{n_seqs:6d} seqs  {n_objs:5d} objects  "
           f"{len(synsets):3d} synsets"
           + (f"  {n_dup} duplicates skipped" if n_dup else ""))

print(f"\nTotal sequences after merge: {len(all_sequences)}")
n_objects = len({(s['synset_id'], s['obj_id']) for s in all_sequences})
print(f"Unique objects:              {n_objects}")

if args.verbose:
    # Breakdown by synset
    from collections import Counter
    synset_counts = Counter(s['synset_id'] for s in all_sequences)
    print(f"\n  Sequences per synset ({len(synset_counts)} synsets):")
    for syn, cnt in sorted(synset_counts.items()):
        print(f"    {syn}  {cnt:7d}")

    # Sequences per object distribution
    obj_seq_counts = Counter(
        (s['synset_id'], s['obj_id']) for s in all_sequences
    )
    counts = list(obj_seq_counts.values())
    print(f"\n  Sequences per object: "
          f"min={min(counts)}  max={max(counts)}  "
          f"mean={sum(counts)/len(counts):.1f}")

    # Active factors distribution
    factor_counts = Counter(
        f for s in all_sequences for f in s['traversal_factors']
    )
    latent_names = reference_header['latent_names']
    print(f"\n  Active factor frequency:")
    for k, name in enumerate(latent_names):
        print(f"    [{k}] {name:12s}  {factor_counts.get(k, 0):7d} sequences")

    # Direction distribution
    dir_counts = Counter(
        int(d)
        for s in all_sequences
        for d in s['traversal_directions']
        if d != 0
    )
    print(f"\n  Traversal directions (non-frozen):  "
          f"+1 (forward): {dir_counts.get(1,0)}  "
          f"-1 (backward): {dir_counts.get(-1,0)}")

merged = {**reference_header, 'sequences': all_sequences}

out_path = os.path.join(args.output_dir, f'{args.out_name}.pkl')
tmp_path = out_path + '.tmp'
with open(tmp_path, 'wb') as f:
    pickle.dump(merged, f)
os.replace(tmp_path, out_path)
print(f"\nMerged metadata written to {out_path}")
