import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(
    description="Split all_objects.npy into per-job subset files for parallel rendering."
)
parser.add_argument('--objects', required=True,
                    help="Path to all_objects.npy (shape [N, 2])")
parser.add_argument('--n-jobs', type=int, required=True,
                    help="Number of subsets to create")
parser.add_argument('--output-dir', default='.',
                    help="Directory to write objects_NNN.npy files (default: .)")
parser.add_argument('--max-per-synset', type=int, default=None,
                    help="Keep at most this many objects per synset (default: unlimited)")
args = parser.parse_args()

items = np.load(args.objects)
print(f"Loaded {len(items)} objects from {args.objects}")

if args.max_per_synset is not None:
    synsets, counts = np.unique(items[:, 0], return_counts=True)
    kept = []
    for synset in synsets:
        mask = items[:, 0] == synset
        kept.append(items[mask][:args.max_per_synset])
    items = np.concatenate(kept, axis=0)
    print(f"After --max-per-synset {args.max_per_synset}: {len(items)} objects "
          f"across {len(synsets)} synsets")

os.makedirs(args.output_dir, exist_ok=True)

chunks = np.array_split(items, args.n_jobs)
for i, chunk in enumerate(chunks):
    out_path = os.path.join(args.output_dir, f'objects_{i:03d}.npy')
    np.save(out_path, chunk)
    print(f"  {out_path}  ({len(chunk)} objects)")

print(f"Done. {args.n_jobs} subset files written to {args.output_dir}")
