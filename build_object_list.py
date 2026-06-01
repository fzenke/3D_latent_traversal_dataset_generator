import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(
    description="Scan a ShapeNet Core V2 directory and build an all_objects.npy file."
)
parser.add_argument('--models-path', required=True,
                    help="Path to ShapeNet Core V2 root (contains synset_id subdirectories)")
parser.add_argument('--output', default='all_objects.npy',
                    help="Output .npy file path (default: all_objects.npy)")
parser.add_argument('--synsets', nargs='+', default=None,
                    help="Restrict to these synset IDs (default: all synsets)")
args = parser.parse_args()

objects = []
synset_dirs = sorted(os.listdir(args.models_path))

for synset_id in synset_dirs:
    if args.synsets and synset_id not in args.synsets:
        continue
    synset_path = os.path.join(args.models_path, synset_id)
    if not os.path.isdir(synset_path):
        continue
    n_before = len(objects)
    for obj_id in sorted(os.listdir(synset_path)):
        obj_path = os.path.join(synset_path, obj_id, 'models', 'model_normalized.obj')
        if os.path.isfile(obj_path):
            objects.append((synset_id, obj_id))
    print(f"  {synset_id}  {len(objects) - n_before} objects")

arr = np.array(objects)
np.save(args.output, arr)
print(f"\nTotal: {len(arr)} objects saved to {args.output}")
