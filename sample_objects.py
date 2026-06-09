import argparse
import os
import sys
from collections import Counter

import numpy as np


def _valid_objects(models_path, synset_id):
    """Return sorted list of obj_ids that have a model_normalized.obj."""
    synset_path = os.path.join(models_path, synset_id)
    if not os.path.isdir(synset_path):
        sys.exit(f"ERROR: synset directory not found: {synset_path}")
    return sorted(
        obj_id for obj_id in os.listdir(synset_path)
        if os.path.isfile(
            os.path.join(synset_path, obj_id, 'models', 'model_normalized.obj')
        )
    )


def collect_objects(models_path, synsets=None, objects=None, seed=0):
    """Build a deduplicated list of (synset_id, obj_id) pairs.

    Args:
        models_path: Path to ShapeNet Core V2 root.
        synsets:     List of (synset_id, n) tuples — sample n random objects each.
        objects:     List of (synset_id, obj_id) tuples — pinned specific objects.
        seed:        Integer RNG seed for deterministic sampling.

    Returns:
        List of (synset_id, obj_id) string pairs, deduplicated.
        Pinned objects take insertion priority; random samples fill in the rest.
    """
    synsets = synsets or []
    objects = objects or []
    rng = np.random.default_rng(seed)

    # Use an ordered dict to deduplicate while preserving insertion order.
    # Explicit objects are added first so they survive dedup.
    selected = {}

    for synset_id, obj_id in objects:
        obj_path = os.path.join(models_path, synset_id, obj_id,
                                'models', 'model_normalized.obj')
        if not os.path.isfile(obj_path):
            sys.exit(f"ERROR: object not found: {synset_id}/{obj_id} ({obj_path})")
        selected[(synset_id, obj_id)] = True

    for synset_id, n in synsets:
        ids = _valid_objects(models_path, synset_id)
        if n is None:
            chosen = ids
        else:
            if n > len(ids):
                sys.exit(
                    f"ERROR: requested {n} objects from {synset_id} "
                    f"but only {len(ids)} available"
                )
            chosen = rng.choice(ids, size=n, replace=False)
        for obj_id in chosen:
            selected.setdefault((synset_id, obj_id), True)

    return list(selected.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Build a targeted objects.npy by sampling random objects per synset "
                    "and/or pinning specific objects by ID."
    )
    parser.add_argument('--models-path', required=True,
                        help="Path to ShapeNet Core V2 root")
    parser.add_argument('--synset', nargs=2, metavar=('SYNSET_ID', 'N'), action='append',
                        default=[],
                        help="Sample N random objects from SYNSET_ID (repeatable). "
                             "Use 'all' for N to include every object in that synset.")
    parser.add_argument('--object', nargs=2, metavar=('SYNSET_ID', 'OBJ_ID'), action='append',
                        default=[],
                        help="Include a specific object (repeatable)")
    parser.add_argument('--output', default='objects.npy',
                        help="Output .npy path (default: objects.npy)")
    parser.add_argument('--seed', type=int, default=0,
                        help="RNG seed for deterministic sampling (default: 0)")
    args = parser.parse_args()

    if not args.synset and not args.object:
        parser.error("Specify at least one --synset or --object entry.")

    synsets = [(s, None if n == 'all' else int(n)) for s, n in args.synset]
    objects = list(map(tuple, args.object))

    pairs = collect_objects(args.models_path, synsets=synsets, objects=objects, seed=args.seed)
    arr = np.array(pairs)
    np.save(args.output, arr)

    counts = Counter(s for s, _ in pairs)
    print(f"Saved {len(arr)} objects to {args.output}")
    for synset_id, cnt in sorted(counts.items()):
        print(f"  {synset_id}  {cnt} objects")


if __name__ == '__main__':
    main()
