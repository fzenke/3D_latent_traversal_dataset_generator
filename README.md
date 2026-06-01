
# 3D Latent Traversal Dataset Generator

> This code is based on the [3DIEBench](https://github.com/facebookresearch/SIE) dataset generator by Garrido et al. (Meta AI), originally released under the MIT License.

Generates latent traversals of 3D objects for studying **invariance and equivariance** in self-supervised learning. Instead of sampling random latents per image, this generator produces *sequences* in which one latent factor is swept linearly across its full range while all others are held fixed — enabling controlled studies of what a model learns to be invariant or equivariant to.

Built on [BlenderProc](https://github.com/DLR-RM/BlenderProc) and [ShapeNet Core V2](https://shapenet.org/).

---

## Latent Space

Each scene is described by 7 continuous factors:

| Index | Name | Range |
|---|---|---|
| 0 | `rot_x` | [−π/2, π/2] |
| 1 | `rot_y` | [−π/2, π/2] |
| 2 | `rot_z` | [−π/2, π/2] |
| 3 | `floor_hue` | [0, 1] |
| 4 | `spot_theta` | [0, π/4] |
| 5 | `spot_phi` | [0, 2π] |
| 6 | `spot_hue` | [0, 1] |

Rotations use Tait-Bryan (XYZ extrinsic) Euler angles.

---

## Dataset Structure

```
{output_dir}/
  metadata.pkl              # all metadata; references relative frame paths
  seqs/
    {synset_id}/
      {obj_id[:2]}/         # 2-char prefix bucket to limit directory width
        {obj_id}/
          seq_00/           # factor 0 (rot_x) traversal
            frame_0000.jpg
            frame_0001.jpg
            ...
          seq_01/           # factor 1 (rot_y) traversal
            ...
          seq_06/           # factor 6 (spot_hue) traversal
            ...
```

Each object produces **7 sequences** (one per factor). Within a sequence, the traversed factor sweeps linearly from its minimum to maximum over T frames; all other factors are held at a shared random base value sampled once per object.

### `metadata.pkl` layout

```python
{
    'latent_names':  list[str],          # 7 factor names
    'latent_ranges': np.ndarray,         # shape [7, 2] — (min, max) per factor
    'n_frames':      int,                # frames per sequence
    'image_size':    int,                # pixel resolution (square)
    'sequences': [
        {
            'synset_id':        str,
            'obj_id':           str,
            'traversal_factor': int,          # 0–6
            'base_latent':      np.ndarray,   # shape [7] — fixed background state
            'latents':          np.ndarray,   # shape [T, 7]
            'frames_dir':       str,          # relative path to sequence directory
        },
        ...
    ]
}
```

---

## Generating the Dataset

### Requirements

- [BlenderProc](https://github.com/DLR-RM/BlenderProc) (`pip install blenderproc`)
- [ShapeNet Core V2](https://shapenet.org/) (requires free registration)

### Building the object list

Before generating, scan your ShapeNet installation to produce `all_objects.npy`:

```bash
python build_object_list.py --models-path /path/to/ShapeNetCoreV2
```

This walks all synset directories, keeps only objects that have a valid `model_normalized.obj`, and writes `all_objects.npy` (an array of `(synset_id, obj_id)` string pairs). To restrict to specific categories, pass their synset IDs:

```bash
python build_object_list.py \
  --models-path /path/to/ShapeNetCoreV2 \
  --synsets 02691156 02958343 03001627 \
  --output my_objects.npy
```

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
| `--output-dir` | `./3D_latent_traversal` | Output root directory |
| `--objects` | required | `.npy` file of `(synset_id, obj_id)` tuples |
| `--image-size` | `256` | Render resolution in pixels (square) |
| `--n-frames` | `32` | Frames per traversal sequence |
| `--seed` | `0` | RNG seed for base latent sampling |
| `--max-objects` | — | Limit number of objects (testing) |
| `--max-sequences` | — | Limit total sequences (testing) |

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

The merge script validates that all partial files share the same header (latent names, ranges, frame count, image size), deduplicates any overlapping sequences, and writes the result atomically. Pass `--out-name` to use a custom output filename.

The generator resumes automatically: any sequence directory that already contains the expected number of frames is skipped. The pickle is updated after each object, so a crash loses at most one object's work.

---

## Loading in PyTorch

```python
from dataset import TraversalDataset
from torch.utils.data import DataLoader

ds = TraversalDataset('3D_latent_traversal/metadata.pkl', '3D_latent_traversal')

batch = ds[0]
# batch['frames']:            FloatTensor  [T, 3, H, W]  — values in [0, 1]
# batch['latents']:           FloatTensor  [T, 7]
# batch['base_latent']:       FloatTensor  [7]
# batch['synset_id']:         str
# batch['obj_id']:            str
# batch['traversal_factor']:  int  (0–6)

# Filter helpers
rot_seqs  = ds.filter_by_factor('rot_x')       # or filter_by_factor(0)
airplanes = ds.filter_by_synset('02691156')

loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
```

A custom `transform` (receives a `PIL.Image`, returns a tensor) can be passed to the constructor to plug in any `torchvision` augmentation pipeline.
