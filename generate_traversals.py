import blenderproc as bproc
import sys
import os
# Ensure the directory containing this script is on sys.path so that
# latent_utils.py (a sibling module) can be found when BlenderProc runs us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import bpy
from mathutils import Matrix, Euler
import numpy as np
import cv2
import matplotlib
import pickle
import signal
import time

from latent_utils import (  # noqa: F401 — re-exported for the rest of this script
    LATENT_NAMES, LATENT_RANGES, N_FACTORS, CIRCULAR_FACTORS,
    _seq_seed, make_traversal, sample_velocities, _elastic_bounce, build_latents,
)


# ====================================================================
#  HELPER FUNCTIONS  (shared with main.py)
# ====================================================================
def set_camera_pose(cam2world_matrix, frame=None):
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)
    cam_ob = bpy.context.scene.camera
    cam_ob.matrix_world = cam2world_matrix
    bpy.context.scene.frame_end = frame + 1
    cam_ob.keyframe_insert(data_path='location', frame=frame)
    cam_ob.keyframe_insert(data_path='rotation_euler', frame=frame)
    return frame


def spherical_to_cartesian(r, theta, phi):
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


def apply_latent(latent, floor_obj, spot_light):
    """Push a 10-element latent vector into the Blender scene."""
    rot_x, rot_y, rot_z, floor_hue, spot_theta, spot_phi, spot_hue, \
        trans_x, trans_y, trans_z = latent

    # Object rotation and translation (applied to bpy active object — see caller)
    angles = (float(rot_x), float(rot_y), float(rot_z))
    location = (float(trans_x), float(trans_y), float(trans_z))

    # Floor colour
    rgb = matplotlib.colors.hsv_to_rgb((floor_hue, 0.6, 0.6))
    floor_obj.active_material.diffuse_color = (*rgb, 1)

    # Spot position + colour
    loc = spherical_to_cartesian(4, spot_theta, spot_phi)
    rot_mat = bproc.camera.rotation_from_forward_vec(np.array([0, 0, 0]) - loc)
    spot_light.blender_obj.matrix_world = Matrix(
        bproc.math.build_transformation_mat(loc, rot_mat)
    )
    rgb = matplotlib.colors.hsv_to_rgb((spot_hue, 1.0, 0.8))
    spot_light.set_color(rgb)

    return angles, location


def _draw_latent_overlay(img_bgr, latents_t, traversal_factors):
    """Return a copy of img_bgr with per-frame latent values rendered as semi-transparent text.

    Varying factors are drawn in cyan; frozen factors in light grey.
    """
    overlay = img_bgr.copy()
    h = img_bgr.shape[0]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.28, h / 900)
    line_h     = max(1, int(h / (N_FACTORS + 2)))
    for k, (name, val) in enumerate(zip(LATENT_NAMES, latents_t)):
        color  = (0, 230, 230) if k in traversal_factors else (160, 160, 160)
        cv2.putText(overlay, f"{name}: {val:.3f}", (4, line_h * (k + 1)),
                    font, font_scale, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)


def seq_is_complete(seq_dir, n_frames):
    """True iff seq_dir exists and contains exactly n_frames JPEGs."""
    if not os.path.isdir(seq_dir):
        return False
    jpegs = [f for f in os.listdir(seq_dir) if f.endswith('.jpg')]
    return len(jpegs) == n_frames


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


class LoadTimeout(Exception):
    pass


def handle_sigalrm(signum, frame):
    raise LoadTimeout()


def _skip_object(synset, obj_id, sentinel_path, skip_list_path, skip_set):
    """Add an object to the runtime skip set, optionally persist to skip list,
    clean up its sentinel file, and remove any partial Blender scene object."""
    key = f"{synset}/{obj_id}"
    skip_set.add(key)
    if skip_list_path:
        with open(skip_list_path, 'a') as _f:
            _f.write(f"{key}\n")
        print(f"  Auto-appended to skip list: {skip_list_path}", flush=True)
    try:
        bpy.data.objects.remove(bpy.context.visible_objects[-1], do_unlink=True)
    except Exception:
        pass
    try:
        os.remove(sentinel_path)
    except FileNotFoundError:
        pass


# ====================================================================
#  CLI
# ====================================================================
parser = argparse.ArgumentParser(
    description="Generate 3DIEBench temporal traversal dataset via BlenderProc"
)
parser.add_argument('--models-path', required=True,
                    help="Path to ShapeNet Core V2 root")
parser.add_argument('--output-dir', required=True, default='./3DIEBench_traversal',
                    help="Output root directory")
parser.add_argument('--objects', required=True,
                    help="Path to .npy file of (synset_id, obj_id) tuples")
parser.add_argument('--image-size', type=int, default=256,
                    help="Render resolution in pixels (square)")
parser.add_argument('--render-samples', type=int, default=50,
                    help="Cycles render samples per frame (default: 50). "
                         "Lower values render faster with more noise; "
                         "combine with a denoiser for best quality/speed trade-off.")
parser.add_argument('--n-frames', type=int, default=32,
                    help="Number of frames per traversal sequence")
parser.add_argument('--seed', type=int, default=0,
                    help="RNG seed for reproducibility")
parser.add_argument('--max-objects', type=int, default=None,
                    help="Cap the number of objects rendered (for testing)")
parser.add_argument('--max-sequences', type=int, default=None,
                    help="Cap the total number of sequences rendered (for testing)")
parser.add_argument('--metadata-name', default='metadata',
                    help="Base name for the metadata pickle file, without .pkl "
                         "(default: metadata). Use e.g. metadata_000 for parallel jobs.")
parser.add_argument('--full-rotation', action='store_true', default=False,
                    help="Extend rotation range to [-π, π] (360°) instead of "
                         "[-π/2, π/2] (180°, default)")
parser.add_argument('--random-offset', action='store_true', default=False,
                    help="Start each traversal at a random phase within the factor "
                         "range rather than always at the minimum value")
parser.add_argument('--multi-factor', action='store_true', default=False,
                    help="Enable multi-factor traversal mode: each sequence varies "
                         "several factors simultaneously with random directions")
parser.add_argument('--freeze-prob', type=float, default=0.5,
                    help="[multi-factor] Probability that any individual factor is "
                         "frozen (does not vary) in a sequence (default: 0.5). "
                         "At least one factor always varies.")
parser.add_argument('--velocity-stdev', type=float, default=0.0,
                    help="Scale of the velocity distribution per active factor "
                         "(default: 0 = ±1, i.e. always one full range sweep). "
                         "For 'gaussian': velocity ~ N(0, stdev). "
                         "For 'uniform': velocity ~ U(-stdev, +stdev). "
                         "stdev=1 means one full range on average. "
                         "Traversals start at the base latent and reflect at boundaries.")
parser.add_argument('--velocity-dist', default='gaussian', choices=['gaussian', 'uniform'],
                    help="Distribution for velocity sampling when --velocity-stdev > 0: "
                         "'gaussian' (default) draws from N(0, velocity_stdev); "
                         "'uniform' draws from U(-velocity_stdev, +velocity_stdev).")
parser.add_argument('--seqs-per-object', type=int, default=7,
                    help="[multi-factor] Number of sequences to generate per object "
                         "(default: 7, matching single-factor mode)")
parser.add_argument('--factors', nargs='+', default=None,
                    help="Restrict traversals to these factors only. "
                         "Accepts names (e.g. rot_x rot_z) or indices (0 2). "
                         "Default: all 7 factors.")
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help="Print per-object and per-sequence debug info including "
                         "latent values, directions, and timing")
parser.add_argument('--skip-list', default=None,
                    help="Path to a text file listing synset_id/obj_id entries "
                         "(one per line) to skip unconditionally, e.g. for models "
                         "that hang during loading")
parser.add_argument('--load-timeout', type=int, default=120,
                    help="Seconds to wait for load_shapenet before treating the "
                         "model as a hang (default: 120, set to 0 to disable). "
                         "Timed-out objects are auto-appended to --skip-list.")
parser.add_argument('--debug-overlay', action='store_true', default=False,
                    help="Burn per-frame latent values onto each saved image for "
                         "visual inspection. Varying factors are shown in cyan, "
                         "frozen factors in grey.")

args = parser.parse_args()


def vprint(*a, **kw):
    if args.verbose:
        print(*a, **kw)

if args.factors is not None:
    active_factors = set()
    for f in args.factors:
        if f.lstrip('-').isdigit():
            active_factors.add(int(f))
        else:
            if f not in LATENT_NAMES:
                raise ValueError(f"Unknown factor name '{f}'. "
                                 f"Valid names: {LATENT_NAMES}")
            active_factors.add(LATENT_NAMES.index(f))
else:
    active_factors = set(range(N_FACTORS))

if args.full_rotation:
    LATENT_RANGES[:3] = np.array([[-np.pi, np.pi]] * 3)

np.random.seed(args.seed)

signal.signal(signal.SIGUSR1, handle_sigusr1)
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGALRM, handle_sigalrm)

# ====================================================================
#  LOAD OBJECTS
# ====================================================================
items = np.load(args.objects)
print(f"Loaded {len(items)} objects from {args.objects}")

skip_set = set()
if args.skip_list and os.path.exists(args.skip_list):
    with open(args.skip_list) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                skip_set.add(line)
    print(f"Skip list loaded: {len(skip_set)} entries from {args.skip_list}")

items = [item for item in items
         if f"{item[0]}/{item[1]}" not in skip_set]
if skip_set:
    print(f"Objects after skip-list filter: {len(items)}")

if args.max_objects is not None:
    items = items[:args.max_objects]
    print(f"Capped to {len(items)} objects (--max-objects)")

if args.multi_factor:
    seq_indices = range(args.seqs_per_object)
else:
    seq_indices = sorted(active_factors)
jobs = [(str(item[0]), str(item[1]), seq_idx)
        for item in items
        for seq_idx in seq_indices]

if args.max_sequences is not None:
    jobs = jobs[:args.max_sequences]
    print(f"Capped to {len(jobs)} sequences (--max-sequences)")

print(f"Total sequences to generate: {len(jobs)}")
vprint(f"Mode:           {'multi-factor' if args.multi_factor else 'single-factor'}")
vprint(f"n_frames:       {args.n_frames}  |  image_size: {args.image_size}  |  render_samples: {args.render_samples}")
vprint(f"full_rotation:  {args.full_rotation}  |  random_offset: {args.random_offset}")
vprint(f"active_factors: {sorted(active_factors)} "
       f"({', '.join(LATENT_NAMES[k] for k in sorted(active_factors))})")
if args.multi_factor:
    vprint(f"freeze_prob:    {args.freeze_prob}  |  seqs_per_object: {args.seqs_per_object}")

# ====================================================================
#  BLENDER SCENE INIT
# ====================================================================
bproc.init()

image_size = args.image_size
n_frames = args.n_frames
distance = 2.5

# Floor plane
bpy.ops.mesh.primitive_plane_add(size=10000, location=(0, 0, -1))
floor = bpy.context.active_object
mat = bpy.data.materials.new(name="FloorMaterial")
floor.data.materials.append(mat)

# Sun
sun = bproc.types.Light()
sun.set_type("SUN")
sun.set_energy(1.5)
sun.blender_obj.data.angle = np.pi / 2

# Spot
spot = bproc.types.Light()
spot.set_type("SPOT")
spot.set_location([0, 0, 2])
spot.set_energy(500)
spot.blender_obj.data.spot_size = np.pi / 8

bproc.renderer.set_max_amount_of_samples(args.render_samples)
bproc.camera.set_resolution(image_size, image_size)

# Fixed camera
camera_theta = np.pi / 4
cam_loc = spherical_to_cartesian(distance, camera_theta, np.pi / 2)
cam_rot = bproc.camera.rotation_from_forward_vec(np.array([0, 0, 0]) - cam_loc)
cam2world = bproc.math.build_transformation_mat(cam_loc, cam_rot)
set_camera_pose(cam2world, frame=0)

# ====================================================================
#  METADATA  (load existing or initialise fresh)
# ====================================================================
os.makedirs(args.output_dir, exist_ok=True)
pkl_path = os.path.join(args.output_dir, f'{args.metadata_name}.pkl')

# Sentinel file: written just before load_shapenet, deleted on success.
# If found at startup it means the previous run hung mid-load on that object.
sentinel_path = os.path.join(args.output_dir, f'.loading_{args.metadata_name}')
if os.path.exists(sentinel_path):
    with open(sentinel_path) as _f:
        _hung = _f.read().strip()
    print(f"WARNING: sentinel found — previous run hung loading '{_hung}', adding to skip set")
    skip_set.add(_hung)
    os.remove(sentinel_path)
    if args.skip_list:
        with open(args.skip_list, 'a') as _f:
            _f.write(f"{_hung}\n")
        print(f"  Auto-appended to skip list: {args.skip_list}")

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Resumed existing metadata ({len(metadata['sequences'])} sequences already recorded)")
else:
    metadata = {
        'latent_names': LATENT_NAMES,
        'latent_ranges': LATENT_RANGES,
        'n_frames': n_frames,
        'image_size': image_size,
        'config': {
            'seed':            args.seed,
            'models_path':     args.models_path,
            'objects':         args.objects,
            'render_samples':  args.render_samples,
            'multi_factor':    args.multi_factor,
            'freeze_prob':     args.freeze_prob,
            'seqs_per_object': args.seqs_per_object,
            'velocity_stdev':  args.velocity_stdev,
            'velocity_dist':   args.velocity_dist,
            'active_factors':  sorted(active_factors),
            'random_offset':   args.random_offset,
            'full_rotation':   args.full_rotation,
        },
        'sequences': [],
    }

# Index already-recorded sequences for fast resume lookup
recorded = {
    (s['synset_id'], s['obj_id'], s['seq_idx'])
    for s in metadata['sequences']
}

# ====================================================================
#  MAIN RENDER LOOP
# ====================================================================
current_synset = None
current_obj_id = None
model_obj = None
base_latent = None

# Group jobs by object so we load each mesh only once
from itertools import groupby
jobs_by_obj = {}
for synset, obj_id, seq_idx in jobs:
    jobs_by_obj.setdefault((synset, obj_id), []).append(seq_idx)

total_objects = len(jobs_by_obj)
obj_count = 0
frames_rendered = 0
frames_skipped = 0
t_run_start = time.time()

first_object = True
for (synset, obj_id), factors in jobs_by_obj.items():
    obj_count += 1

    # Skip objects in the runtime skip set (sentinel catches + in-process timeouts)
    if f"{synset}/{obj_id}" in skip_set:
        vprint(f"[{obj_count}/{total_objects}] {synset}/{obj_id} — in skip set, skipping")
        continue

    # Check if all sequences for this object are already done
    if all((synset, obj_id, seq_idx) in recorded for seq_idx in factors):
        vprint(f"[{obj_count}/{total_objects}] {synset}/{obj_id} — all seqs already recorded, skipping")
        continue

    # Load (or replace) the ShapeNet model
    t_load = time.time()
    print(f"[{obj_count}/{total_objects}] {synset}/{obj_id} — loading mesh ...", flush=True)
    if not first_object:
        vprint(f"  removing previous object from scene", flush=True)
        bpy.data.objects.remove(bpy.context.visible_objects[-1], do_unlink=True)
        vprint(f"  removed  ({time.time()-t_load:.2f}s)", flush=True)
    first_object = False

    # --- hang detection: sentinel + optional SIGALRM timeout ---
    with open(sentinel_path, 'w') as _f:
        _f.write(f"{synset}/{obj_id}")
    if args.load_timeout:
        signal.alarm(args.load_timeout)

    try:
        model_obj = bproc.loader.load_shapenet(
            args.models_path, used_synset_id=synset, used_source_id=obj_id
        )
    except LoadTimeout:
        signal.alarm(0)
        print(f"  TIMEOUT: load_shapenet hung >{args.load_timeout}s on "
              f"{synset}/{obj_id} — skipping", flush=True)
        _skip_object(synset, obj_id, sentinel_path, args.skip_list, skip_set)
        first_object = True
        continue
    except Exception as _e:
        signal.alarm(0)
        print(f"  ERROR loading {synset}/{obj_id}: {_e} — skipping", flush=True)
        _skip_object(synset, obj_id, sentinel_path, args.skip_list, skip_set)
        first_object = True
        continue

    if args.load_timeout:
        signal.alarm(0)
    os.remove(sentinel_path)
    # -----------------------------------------------------------

    vprint(f"  load_shapenet done  ({time.time()-t_load:.2f}s)", flush=True)

    vprint(f"  get_bound_box ...", flush=True)
    bb = model_obj.get_bound_box()
    bb_center = np.mean(bb, axis=0)
    vprint(f"  bb_center={bb_center}  ({time.time()-t_load:.2f}s)", flush=True)

    vprint(f"  set_origin ...", flush=True)
    model_obj.set_origin(point=bb_center)
    vprint(f"  set_origin done  ({time.time()-t_load:.2f}s)", flush=True)

    # Apply smooth shading to every mesh object loaded from ShapeNet.
    # ShapeNet OBJ files embed explicit vertex normals (vn) which Blender stores
    # as custom split normals; these silently override any smooth/flat setting and
    # must be cleared so Blender recomputes normals from the geometry.
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.shade_smooth_by_angle(angle=np.radians(60))
        obj.select_set(False)

    vprint(f"  set_location ...", flush=True)
    model_obj.set_location((0, 0, 0))
    vprint(f"  set_location done  ({time.time()-t_load:.2f}s)", flush=True)

    vprint(f"  ready  (total load: {time.time()-t_load:.2f}s)")

    # Base values for excluded factors: midpoint of range (canonical, not random)
    fixed_base = np.array([(lo + hi) / 2 for lo, hi in LATENT_RANGES])

    for seq_idx in factors:
        if (synset, obj_id, seq_idx) in recorded:
            vprint(f"  seq_{seq_idx:02d} — already recorded, skipping")
            continue

        # Fresh deterministic base latent per sequence (active factors only)
        base_rng = np.random.default_rng(_seq_seed(args.seed, synset, obj_id, 'base', seq_idx))
        base_latent = np.array([
            base_rng.uniform(lo, hi) for lo, hi in LATENT_RANGES
        ])
        for k in range(N_FACTORS):
            if k not in active_factors:
                base_latent[k] = fixed_base[k]
        vprint(f"  seq_{seq_idx:02d} base: " + "  ".join(
            f"{LATENT_NAMES[k]}={base_latent[k]:.3f}" for k in range(N_FACTORS)
        ))

        # Determine per-factor velocities
        if args.multi_factor:
            velocities = sample_velocities(
                args.freeze_prob, args.velocity_stdev, args.velocity_dist,
                args.seed, synset, obj_id, seq_idx,
                allowed_factors=active_factors,
            )
        else:
            # Single-factor mode: seq_idx IS the factor index
            velocities = np.zeros(N_FACTORS, dtype=np.float32)
            if args.velocity_stdev > 0:
                rng = np.random.default_rng(_seq_seed(args.seed, synset, obj_id, 'vel', seq_idx))
                if args.velocity_dist == 'uniform':
                    velocities[seq_idx] = float(rng.uniform(-args.velocity_stdev, args.velocity_stdev))
                else:
                    velocities[seq_idx] = float(rng.standard_normal() * args.velocity_stdev)
            else:
                velocities[seq_idx] = 1.0

        traversal_factors = [int(k) for k in np.where(velocities != 0.0)[0]]

        # Sequence output directory
        seq_rel = os.path.join('seqs', synset, obj_id[:2], obj_id, f'seq_{seq_idx:04d}')
        seq_dir = os.path.join(args.output_dir, seq_rel)

        latents = build_latents(base_latent, n_frames, velocities)

        if seq_is_complete(seq_dir, n_frames):
            print(f"  Skipping {seq_rel} — frames already on disk")
            frames_skipped += n_frames
        else:
            factor_str = '+'.join(LATENT_NAMES[k] for k in traversal_factors)
            print(f"  Rendering {seq_rel}  (factors={factor_str})")

            vprint(f"  Velocities: {[f'{v:+.3f}' for v in velocities]}")
            for k in traversal_factors:
                v = velocities[k]
                vprint(f"    {LATENT_NAMES[k]:12s}  "
                       f"{latents[0,k]:.3f} → {latents[-1,k]:.3f}  "
                       f"(v={v:+.3f})")

            os.makedirs(seq_dir, exist_ok=True)
            bpy_obj = bpy.context.visible_objects[-1]
            t_seq = time.time()
            for t in range(n_frames):
                angles, location = apply_latent(latents[t], floor, spot)
                bpy_obj.rotation_euler = Euler(angles)
                bpy_obj.location = location

                data = bproc.renderer.render()
                frame_path = os.path.join(seq_dir, f'frame_{t:04d}.jpg')
                frame_img = data["colors"][0]
                if args.debug_overlay:
                    frame_img = _draw_latent_overlay(frame_img, latents[t], traversal_factors)
                cv2.imwrite(frame_path, frame_img)
                frames_rendered += 1

            seq_elapsed = time.time() - t_seq
            fps = n_frames / seq_elapsed if seq_elapsed > 0 else 0

            # ETA based on running average fps
            total_frames_target = len(jobs) * n_frames
            frames_done = frames_rendered + frames_skipped
            frames_remaining = total_frames_target - frames_done
            overall_fps = frames_rendered / (time.time() - t_run_start) if frames_rendered > 0 else fps
            eta_s = frames_remaining / overall_fps if overall_fps > 0 else 0
            eta_h, eta_m = divmod(int(eta_s) // 60, 60)
            vprint(f"    seq done: {seq_elapsed:.1f}s  ({fps:.2f} fps)  |  "
                   f"ETA {eta_h}h {eta_m:02d}m  "
                   f"({frames_rendered} rendered, {frames_skipped} skipped)")

        metadata['sequences'].append({
            'synset_id': synset,
            'obj_id': obj_id,
            'seq_idx': seq_idx,
            'traversal_factors': traversal_factors,
            'traversal_velocities': velocities,
            'base_latent': base_latent.copy(),
            'latents': latents,
            'frames_dir': seq_rel,
        })
        recorded.add((synset, obj_id, seq_idx))

    # Atomic write: dump to a temp file then rename so a crash mid-write
    # never leaves a corrupted pickle (os.replace is atomic on POSIX).
    tmp_path = pkl_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(metadata, f)
    os.replace(tmp_path, pkl_path)

total_elapsed = time.time() - t_run_start
overall_fps = frames_rendered / total_elapsed if total_elapsed > 0 and frames_rendered > 0 else 0
print(f"Done. {len(metadata['sequences'])} sequences written to {args.output_dir}")
print(f"Metadata saved to {pkl_path}")
vprint(f"Frames rendered: {frames_rendered}  skipped: {frames_skipped}  "
       f"total time: {total_elapsed/3600:.2f}h  avg fps: {overall_fps:.2f}")
