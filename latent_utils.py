"""Pure latent-space utilities shared by generate_traversals.py and the test suite.

No Blender / BlenderProc imports — safe to use outside a blenderproc context.
"""
import hashlib

import numpy as np

# ====================================================================
#  LATENT SPACE DEFINITION
# ====================================================================
LATENT_NAMES = ['rot_x', 'rot_y', 'rot_z', 'floor_hue',
                'spot_theta', 'spot_phi', 'spot_hue',
                'trans_x', 'trans_y', 'trans_z']
LATENT_RANGES = np.array([
    [-np.pi / 6, np.pi / 6],  # rot_x
    [-np.pi / 6, np.pi / 6],  # rot_y
    [-np.pi, np.pi],           # rot_z
    [0.0, 1.0],                # floor_hue
    [0.0, np.pi / 4],          # spot_theta
    [0.0, 2 * np.pi],          # spot_phi
    [0.0, 1.0],                # spot_hue
    [-0.5, 0.5],               # trans_x
    [-0.5, 0.5],               # trans_y
    [-0.5, 0.5],               # trans_z
], dtype=np.float64)           # shape [10, 2]

N_FACTORS = len(LATENT_NAMES)

# Factors whose values wrap around (hue circle, azimuth angle, full rotation).
# These always use endpoint=False so the sequence tiles without duplicating
# the boundary frame.
CIRCULAR_FACTORS = {2, 3, 5, 6}  # rot_z, floor_hue, spot_phi, spot_hue


def _seq_seed(*parts):
    """Stable 32-bit seed from an arbitrary tuple of identifying parts.

    Using a deterministic hash (rather than the global RNG) ensures that
    resume — skipping already-rendered sequences — never shifts the state
    used by subsequent sequences.
    """
    key = "_".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], 'little')


def make_traversal(k, n_frames, use_random_offset=False,
                   global_seed=0, synset='', obj_id=''):
    """Return a 1-D array of n_frames values for latent factor k.

    Circular factors (hues, spot_phi) always use endpoint=False so the
    sequence wraps cleanly.  When use_random_offset is True every sequence
    starts at a random phase derived deterministically from its identity,
    and wraps at the factor boundary (non-smooth jump accepted for
    non-circular factors).
    """
    lo, hi = LATENT_RANGES[k]
    span = hi - lo
    if use_random_offset:
        rng = np.random.default_rng(_seq_seed(global_seed, synset, obj_id, k))
        offset = rng.uniform(0, span)
    else:
        offset = 0.0
    if k in CIRCULAR_FACTORS or use_random_offset:
        raw = offset + np.linspace(0, span, n_frames, endpoint=False)
        return lo + (raw % span)
    return np.linspace(lo, hi, n_frames)


def sample_velocities(freeze_prob, velocity_stdev, velocity_dist, global_seed, synset, obj_id,
                      seq_idx, allowed_factors=None):
    """Sample a velocity vector for a multi-factor traversal sequence.

    Returns a float32 array of shape [N_FACTORS]:
      0.0   → factor is frozen at its base value
      ±v    → factor sweeps v * span from base_latent in that direction,
               clipped at range boundaries (wrapped for circular factors)

    When velocity_stdev == 0: active factors get ±1.0 (full range sweep).
    When velocity_stdev > 0 and velocity_dist == 'gaussian': N(0, velocity_stdev).
    When velocity_stdev > 0 and velocity_dist == 'uniform':  U(-velocity_stdev, +velocity_stdev).
    Each factor independently freezes with probability freeze_prob.
    Factors not in allowed_factors are always 0.
    At least one allowed factor is guaranteed non-zero.
    """
    if allowed_factors is None:
        allowed_factors = set(range(N_FACTORS))
    rng = np.random.default_rng(_seq_seed(global_seed, synset, obj_id, 'dirs', seq_idx))
    while True:
        active = rng.random(N_FACTORS) >= freeze_prob
        if velocity_stdev > 0:
            if velocity_dist == 'uniform':
                raw = rng.uniform(-velocity_stdev, velocity_stdev, N_FACTORS)
            else:
                raw = rng.standard_normal(N_FACTORS) * velocity_stdev
        else:
            raw = np.where(rng.random(N_FACTORS) < 0.5, 1.0, -1.0)
        velocities = np.where(active, raw, 0.0).astype(np.float32)
        for k in range(N_FACTORS):
            if k not in allowed_factors:
                velocities[k] = 0.0
        if np.any(velocities != 0.0):
            return velocities


def _elastic_bounce(values, lo, hi):
    """Reflect values off [lo, hi] boundaries like an elastic collision."""
    span = hi - lo
    shifted = np.asarray(values, dtype=np.float64) - lo
    modulo = shifted % (2 * span)
    return lo + np.where(modulo <= span, modulo, 2 * span - modulo)


def build_latents(base_latent, n_frames, velocities,
                  velocity_momentum=1.0, rng=None):
    """Build the [n_frames, N_FACTORS] latent matrix for one sequence.

    For each factor k:
      velocities[k] == 0  → all frames use base_latent[k]  (factor stays frozen)
      velocities[k] != 0  → linspace from base_latent[k] by velocity * span;
                             circular factors wrap modulo span,
                             all others reflect elastically off boundaries.

    When velocity_momentum < 1.0 and rng is not None, the velocity of each
    non-frozen factor (v₀ != 0) evolves each frame via an AR(1) random walk:

        v_t = momentum · v_{t-1}  +  √(1 − momentum²) · ε_t,   ε_t ~ N(0, 1)

    Frozen factors (v₀ == 0) are left untouched at base_latent[k], so factors
    excluded via --factors or frozen for this sequence never drift.
    Position is integrated step-by-step; boundaries are handled identically to
    the constant-velocity case.  momentum=1.0 (default) recovers the original
    linear behaviour exactly.
    """
    latents = np.tile(base_latent, (n_frames, 1))
    use_walk = (velocity_momentum < 1.0) and (rng is not None)

    if use_walk:
        noise_scale = np.sqrt(1.0 - velocity_momentum ** 2)
        step_frac = 1.0 / max(n_frames - 1, 1)   # one velocity unit = full span
        for k, v0 in enumerate(velocities):
            if v0 == 0.0:
                continue          # frozen factor (CLI-excluded or per-seq frozen) → stays at base
            lo, hi = LATENT_RANGES[k]
            span = hi - lo
            v = float(v0)
            x = float(base_latent[k])
            traj = np.empty(n_frames)
            for t in range(n_frames):
                traj[t] = x
                v = velocity_momentum * v + noise_scale * float(rng.standard_normal())
                x += v * span * step_frac
            if k in CIRCULAR_FACTORS:
                latents[:, k] = lo + (traj - lo) % span
            else:
                latents[:, k] = _elastic_bounce(traj, lo, hi)
    else:
        for k, v in enumerate(velocities):
            if v == 0.0:
                continue
            lo, hi = LATENT_RANGES[k]
            span = hi - lo
            end = base_latent[k] + v * span
            raw = np.linspace(base_latent[k], end, n_frames)
            if k in CIRCULAR_FACTORS:
                latents[:, k] = lo + (raw - lo) % span
            else:
                latents[:, k] = _elastic_bounce(raw, lo, hi)
    return latents
