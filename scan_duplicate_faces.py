"""Scan ShapeNet .obj files for duplicate faces, without Blender.

Many ShapeNet models contain every face twice, duplicated onto the same shared
vertex set. Every ray hit then ties on depth and the winner flips with BVH
traversal order as the object rotates, producing polygon-bounded patches that
switch on and off between frames. `generate_traversals.py --dedupe-faces`
repairs this at load time; this script measures how widespread it is.

Pure Python — parses the .obj text directly, so it runs over the whole corpus in
minutes without launching Blender. It shares
`mesh_utils.duplicate_face_indices()` with the renderer, so the counts reported
here are produced by the same logic that does the repair.

    python3 scan_duplicate_faces.py --models-path PATH_TO_SHAPENET_CORE_V2 \
        --objects ./my_objects.npy --out dupe_scan.csv --jobs 16

Caveat: indices here are the .obj file's own vertex indices. Blender's importer
can split vertices (on UV or material seams), so its post-import counts may
differ slightly from these. The duplicate-face ratio is the robust signal, not
the absolute vertex count.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

from mesh_utils import duplicate_face_indices, euler_face_budget


def parse_obj_faces(path):
    """Return (n_verts, [face_vertex_indices, ...]) from an .obj file.

    Only 'v' and 'f' lines matter. Face corners look like 'v', 'v/vt', 'v//vn'
    or 'v/vt/vn', so we take the part before the first slash. Negative indices
    are relative to the vertices seen so far, per the .obj spec.
    """
    n_verts = 0
    faces = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            if line.startswith('v '):
                n_verts += 1
            elif line.startswith('f '):
                idx = []
                for corner in line.split()[1:]:
                    tok = corner.split('/', 1)[0]
                    try:
                        i = int(tok)
                    except ValueError:
                        continue
                    idx.append(i if i > 0 else n_verts + 1 + i)
                if len(idx) >= 3:
                    faces.append(idx)
    return n_verts, faces


def scan_model(models_path, synset_id, obj_id):
    """Measure one model. Returns a dict row, or None if the .obj is missing."""
    path = os.path.join(models_path, synset_id, obj_id, 'models',
                        'model_normalized.obj')
    if not os.path.isfile(path):
        return None
    try:
        n_verts, faces = parse_obj_faces(path)
    except OSError as e:
        return {'synset_id': synset_id, 'obj_id': obj_id, 'error': str(e),
                'n_verts': 0, 'n_faces': 0, 'n_dupes': 0, 'dupe_frac': 0.0,
                'over_budget': 0}
    n_faces = len(faces)
    n_dupes = len(duplicate_face_indices(faces))
    budget = euler_face_budget(n_verts, genus=1) if n_verts else 0
    return {
        'synset_id': synset_id,
        'obj_id': obj_id,
        'n_verts': n_verts,
        'n_faces': n_faces,
        'n_dupes': n_dupes,
        'dupe_frac': round(n_dupes / n_faces, 4) if n_faces else 0.0,
        # Faces beyond what the vertex set can support as a clean surface.
        'over_budget': 1 if (budget and n_faces > budget) else 0,
        'error': '',
    }


def _job(task):
    return scan_model(*task)


def load_objects(models_path, objects_npy):
    """(synset_id, obj_id) pairs from an .npy list, or by walking the corpus."""
    if objects_npy:
        import numpy as np
        arr = np.load(objects_npy)
        return [(str(s), str(o)) for s, o in arr]
    pairs = []
    for synset in sorted(os.listdir(models_path)):
        p = os.path.join(models_path, synset)
        if not os.path.isdir(p) or not synset.isdigit():
            continue
        for obj_id in sorted(os.listdir(p)):
            if os.path.isdir(os.path.join(p, obj_id)):
                pairs.append((synset, obj_id))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models-path', required=True,
                    help="ShapeNetCore.v2 root")
    ap.add_argument('--objects', default=None,
                    help="Optional .npy of (synset_id, obj_id) pairs. Default: "
                         "walk every synset under --models-path.")
    ap.add_argument('--out', default='dupe_scan.csv',
                    help="CSV output path (default: dupe_scan.csv)")
    ap.add_argument('--jobs', type=int, default=1,
                    help="Parallel worker processes (default: 1)")
    ap.add_argument('--limit', type=int, default=None,
                    help="Only scan the first N models")
    ap.add_argument('--threshold', type=float, default=0.4,
                    help="dupe_frac above which a model counts as affected "
                         "(default: 0.4; a wholly doubled mesh scores 0.5)")
    args = ap.parse_args()

    if not os.path.isdir(args.models_path):
        sys.exit(f"ERROR: models path not found: {args.models_path}")

    pairs = load_objects(args.models_path, args.objects)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Scanning {len(pairs)} models ...", flush=True)

    tasks = [(args.models_path, s, o) for s, o in pairs]
    rows, missing = [], 0
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.imap_unordered(_job, tasks, chunksize=32)
            for i, row in enumerate(results, 1):
                if row is None:
                    missing += 1
                else:
                    rows.append(row)
                if i % 2000 == 0:
                    print(f"  {i}/{len(tasks)}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            row = _job(t)
            if row is None:
                missing += 1
            else:
                rows.append(row)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    rows.sort(key=lambda r: (-r['dupe_frac'], r['synset_id'], r['obj_id']))
    fields = ['synset_id', 'obj_id', 'n_verts', 'n_faces', 'n_dupes',
              'dupe_frac', 'over_budget', 'error']
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(rows)
    if not n:
        sys.exit("No models scanned.")
    affected = [r for r in rows if r['dupe_frac'] >= args.threshold]
    doubled = [r for r in rows if r['n_faces'] and r['n_dupes'] * 2 == r['n_faces']]
    over = [r for r in rows if r['over_budget']]

    print(f"\nScanned {n} models" + (f", {missing} missing .obj" if missing else ""))
    print(f"  affected (dupe_frac >= {args.threshold}): {len(affected)} "
          f"({100.0 * len(affected) / n:.1f}%)")
    print(f"  exactly doubled (every face twice):       {len(doubled)} "
          f"({100.0 * len(doubled) / n:.1f}%)")
    print(f"  over Euler face budget:                   {len(over)} "
          f"({100.0 * len(over) / n:.1f}%)")

    per_synset = defaultdict(lambda: [0, 0])
    for r in rows:
        per_synset[r['synset_id']][0] += 1
        if r['dupe_frac'] >= args.threshold:
            per_synset[r['synset_id']][1] += 1
    ranked = sorted(per_synset.items(), key=lambda kv: -kv[1][1] / kv[1][0])
    print(f"\nWorst synsets (share of models affected):")
    for synset, (total, bad) in ranked[:15]:
        if not bad:
            break
        print(f"  {synset}  {bad:5d}/{total:<5d}  {100.0 * bad / total:5.1f}%")

    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
