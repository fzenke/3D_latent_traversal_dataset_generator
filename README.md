
# 3D Latent Traversal Dataset Generator

Generates latent traversals of 3D objects for studying **invariance and equivariance** in predictive self-supervised learning. Instead of sampling random latents per image, this generator produces *sequences* in which one or more latent factors sweep across their range while others are held fixed. This strategy enables controlled studies of what a model learns to be invariant or equivariant to.

> This code is based on the [3DIEBench](https://github.com/facebookresearch/SIE) dataset generator by Garrido et al. (Meta AI), originally released under the GPL v3.0 License.


Built on [BlenderProc](https://github.com/DLR-RM/BlenderProc) and [ShapeNet Core V2](https://shapenet.org/).

---

## Latent Space

Each scene is described by 10 continuous factors:

| Index | Name | Default Range |
|---|---|---|
| 0 | `rot_x` | [−π/6, π/6] |
| 1 | `rot_y` | [−π/6, π/6] |
| 2 | `rot_z` | [−π, π] |
| 3 | `floor_hue` | [0, 1] |
| 4 | `spot_theta` | [0, π/4] |
| 5 | `spot_phi` | [0, 2π] |
| 6 | `spot_hue` | [0, 1] |
| 7 | `trans_x` | [−0.5, 0.5] |
| 8 | `trans_y` | [−0.5, 0.5] |
| 9 | `trans_z` | [−0.5, 0.5] |

Rotations use Tait-Bryan (XYZ extrinsic) Euler angles. Pass `--full-rotation` to expand all rotation ranges to [−π, π].

---

## Dataset Structure

```
{output_dir}/
  metadata.pkl              # all metadata; references relative frame paths
  seqs/
    {synset_id}/
      {obj_id[:2]}/         # 2-char prefix bucket to limit directory width
        {obj_id}/
          seq_0000/         # first sequence for this object
            frame_0000.jpg
            frame_0001.jpg
            ...
          seq_0001/         # second sequence
            ...
```

Each object produces `--seqs-per-object` sequences (default 10). In single-factor mode each sequence sweeps exactly one factor; in `--multi-factor` mode multiple factors can vary per sequence.

### `metadata.pkl` layout

```python
{
    'latent_names':  list[str],          # 10 factor names
    'latent_ranges': np.ndarray,         # shape [10, 2] — (min, max) per factor
    'n_frames':      int,                # frames per sequence
    'image_size':    int,                # pixel resolution (square)
    'config': {                          # generation args for reproducibility
        'seed': int,
        'models_path': str,
        'objects': str,
        'render_samples': int,
        'multi_factor': bool,
        'freeze_prob': float,
        'seqs_per_object': int,
        'velocity_stdev': float,
        'velocity_dist': str,
        'active_factors': list[int],
        'random_offset': bool,
        'full_rotation': bool,
    },
    'sequences': [
        {
            'synset_id':            str,
            'obj_id':               str,
            'seq_idx':              int,
            'traversal_factors':    list[int],    # indices of varying factors (0–9)
            'traversal_velocities': np.ndarray,   # shape [10] — 0 = frozen, ±v = sweep speed
            'base_latent':          np.ndarray,   # shape [10] — fixed background state
            'latents':              np.ndarray,   # shape [T, 10]
            'frames_dir':           str,          # relative path to sequence directory
        },
        ...
    ]
}
```

When using `merge_metadata.py` the top-level `config` key becomes `configs: list[dict]` (one entry per partial job).

---

## Generating the Dataset

### Requirements

- [BlenderProc](https://github.com/DLR-RM/BlenderProc) (`pip install blenderproc`)
- [ShapeNet Core V2](https://shapenet.org/) (requires free registration)

### Building the object list

**Option A — include all objects (or filter by synset)**

Scan your ShapeNet installation to produce `all_objects.npy`:

```bash
python build_object_list.py --models-path /path/to/ShapeNetCoreV2
```

This walks all synset directories, keeps only objects that have a valid `model_normalized.obj`, and writes `all_objects.npy`. To restrict to specific categories, pass their synset IDs:

```bash
python build_object_list.py \
  --models-path /path/to/ShapeNetCoreV2 \
  --synsets 02691156 02958343 03001627 \
  --output my_objects.npy
```

**Option B — targeted sampling with `sample_objects.py`**

Sample a fixed number of objects per synset and/or pin specific objects by ID:

```bash
python sample_objects.py \
  --models-path /path/to/ShapeNetCoreV2 \
  --synset 02691156 50 \         # 50 random airplanes
  --synset 02958343 all \        # all cars
  --object 03001627 someObjId \  # one pinned chair
  --output my_objects.npy \
  --seed 0
```

Both scripts write `(synset_id, obj_id)` string pairs to the `.npy` file.

### Full run

```bash
blenderproc run generate_traversals.py \
  --models-path /path/to/ShapeNetCoreV2 \
  --output-dir ./3D_latent_traversal \
  --objects ./all_objects.npy \
  --n-frames 32 \
  --image-size 256 \
  --seed 0
```

### Smoke test (2 objects, 4 frames, small images)

```bash
blenderproc run generate_traversals.py \
  --models-path /path/to/ShapeNetCoreV2 \
  --output-dir ./test_out \
  --objects ./all_objects.npy \
  --max-objects 2 \
  --n-frames 4 \
  --image-size 64
```

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--models-path` | required | Path to ShapeNet Core V2 root |
| `--output-dir` | `./3DIEBench_traversal` | Output root directory |
| `--objects` | required | `.npy` file of `(synset_id, obj_id)` tuples |
| `--image-size` | `256` | Render resolution in pixels (square) |
| `--render-samples` | `50` | Cycles render samples per frame |
| `--n-frames` | `32` | Frames per traversal sequence |
| `--seed` | `0` | RNG seed for base latent sampling |
| `--seqs-per-object` | `10` | Number of traversal sequences per object |
| `--factors` | all | Restrict active latent factors (by name, e.g. `rot_x rot_z`) |
| `--multi-factor` | off | Allow multiple factors to vary within one sequence |
| `--freeze-prob` | `0.5` | Per-factor freeze probability in multi-factor mode |
| `--velocity-stdev` | `0.0` | Velocity std dev (0 = full-range ±1 sweep) |
| `--velocity-dist` | `gaussian` | Velocity distribution: `gaussian` or `uniform` (half-width = `--velocity-stdev`) |
| `--full-rotation` | off | Expand all rotation ranges to [−π, π] |
| `--random-offset` | off | Start traversals at a random phase instead of range minimum |
| `--max-objects` | — | Limit number of objects processed (testing) |
| `--max-sequences` | — | Limit total sequences rendered (testing) |
| `--metadata-name` | `metadata` | Stem for the output pickle file |
| `--skip-list` | — | Text file of `synset_id/obj_id` entries to skip |
| `--load-timeout` | `120` | Seconds before aborting a slow object load |
| `--debug-overlay` | off | Burn per-frame latent values onto saved images |
| `--verbose` / `-v` | off | Print per-sequence progress |

### Parallel runs on a SLURM cluster

**Step 1 — Split the object list**

```bash
python split_objects.py \
  --objects all_objects.npy \
  --n-jobs 21 \
  --output-dir splits
```

This writes `splits/objects_000.npy` … `splits/objects_020.npy`. Each file contains a balanced subset of `(synset_id, obj_id)` pairs. To create a smaller debug dataset, add `--max-per-synset N` to keep at most N objects per category before splitting.

**Step 2 — Submit the array job**

Edit `submit_array.sh` to set `SHAPENET_PATH`, `OUTPUT_DIR`, `SPLITS_DIR`, and `--array=0-<n_jobs-1>`, then submit:

```bash
sbatch submit_array.sh
```

Each SLURM task renders one subset and writes its own `metadata_NNN.pkl` to `OUTPUT_DIR`. Jobs resume automatically on restart: frames already on disk are skipped, and objects that hang or fail during loading are appended to `skip-list.txt` and skipped in future runs.

**Step 3 — Merge metadata**

Once all array tasks have completed, merge the per-job pickle files into a single `metadata.pkl`:

```bash
python merge_metadata.py \
  --output-dir /path/to/OUTPUT_DIR \
  --verbose
```

The merge script validates that all partial files share the same header (latent names, ranges, frame count, image size), deduplicates any overlapping sequences, and writes the result atomically. Pass `--out-name` to use a custom output filename. The merged pickle contains a `configs` key (list of per-job config dicts) for full reproducibility.

The generator resumes automatically: any sequence directory that already contains the expected number of frames is skipped. The pickle is updated after each object, so a crash loses at most one object's work.

---

## Packaging as WebDataset

For large datasets on NFS or other network-mounted storage, packing into
[WebDataset](https://github.com/webdataset/webdataset) tar shards avoids the
small-files bottleneck and enables streaming directly from object storage.

### Pack shards

```bash
pip install webdataset

python package_wds.py \
  --dataset-dir ./3D_latent_traversal \
  --output-dir  ./3D_latent_traversal_wds \
  --shard-maxcount 512
```

Each shard is a plain tar file. One WebDataset *sample* = one full sequence:
T JPEG frames stored as raw bytes (no re-encode), numpy arrays for latents,
and a JSON metadata blob. A `dataset_info.json` is written alongside the
shards so the loader is self-contained (no `metadata.pkl` needed).

Pass `--split train` to prefix shard names (e.g. `train-shard-00000.tar`).

**To produce a train/val split**, pass `--val-fraction`:

```bash
python package_wds.py \
  --dataset-dir ./3D_latent_traversal \
  --output-dir  ./3D_latent_traversal_wds \
  --val-fraction 0.1 \
  --split-seed 42
```

This writes to `output-dir/train/` and `output-dir/val/`, each containing
its own shards and `dataset_info.json`. The split is a random shuffle of
sequences (not objects) using `--split-seed` for reproducibility.

### Load in PyTorch

```python
from wds_dataset import TraversalWebDataset, make_loader

ds     = TraversalWebDataset("./shards/shard-{00000..00020}.tar")
loader = make_loader(ds, batch_size=16, num_workers=4)

for batch in loader:
    # batch['frames']               FloatTensor [B, T, 3, H, W]
    # batch['latents']              FloatTensor [B, T, 10]
    # batch['base_latent']          FloatTensor [B, 10]
    # batch['traversal_velocities'] FloatTensor [B, 10]
    # batch['synset_id']            list[str]
    # batch['obj_id']               list[str]
    # batch['traversal_factors']    list[list[int]]
    ...

# Filter helpers (return new wds pipelines)
rot_seqs  = ds.filter_by_factor('rot_x')   # or filter_by_factor(0)
airplanes = ds.filter_by_synset('02691156')
```

A custom per-frame `transform` (callable `[3, H, W] → [3, H, W]`) can be
passed to `TraversalWebDataset(urls, transform=...)`.

Smoke-test a shard directory:

```bash
python wds_dataset.py --shards "./shards/shard-*.tar" --n-batches 3
```

---

## Loading in PyTorch

```python
from dataset import TraversalDataset
from torch.utils.data import DataLoader

ds = TraversalDataset('3D_latent_traversal/metadata.pkl', '3D_latent_traversal')

batch = ds[0]
# batch['frames']:                FloatTensor  [T, 3, H, W]  — values in [0, 1]
# batch['latents']:               FloatTensor  [T, 10]
# batch['base_latent']:           FloatTensor  [10]
# batch['traversal_velocities']:  FloatTensor  [10]  — 0 = frozen, ±v = sweep speed
# batch['synset_id']:             str
# batch['obj_id']:                str
# batch['traversal_factors']:     list[int]    — indices of varying factors (0–9)

# Filter helpers
rot_seqs  = ds.filter_by_factor('rot_x')       # or filter_by_factor(0)
airplanes = ds.filter_by_synset('02691156')

loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
```

A custom `transform` (receives a `PIL.Image`, returns a tensor) can be passed to the constructor to plug in any `torchvision` augmentation pipeline.
