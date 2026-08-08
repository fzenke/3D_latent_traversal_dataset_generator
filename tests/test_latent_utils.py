import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import latent_utils
from latent_utils import (
    LATENT_NAMES, LATENT_RANGES, N_FACTORS, CIRCULAR_FACTORS,
    _seq_seed, _elastic_bounce, build_latents, sample_velocities,
    enable_full_rotation, ROTATION_FACTORS,
)


# ── _seq_seed ────────────────────────────────────────────────────────────────

def test_seq_seed_deterministic():
    assert _seq_seed(0, 'syn', 'obj', 'dirs', 3) == _seq_seed(0, 'syn', 'obj', 'dirs', 3)

def test_seq_seed_distinct():
    seeds = {_seq_seed(i, 'syn', 'obj') for i in range(10)}
    assert len(seeds) == 10

def test_seq_seed_returns_int():
    s = _seq_seed(42, 'abc', 'xyz')
    assert isinstance(s, int)
    assert 0 <= s < 2**32


# ── _elastic_bounce ──────────────────────────────────────────────────────────

def test_elastic_bounce_in_range():
    result = _elastic_bounce([0.0, 0.5, 1.0], 0.0, 1.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

def test_elastic_bounce_single_reflect_hi():
    # 1.3 is 0.3 past hi=1 → bounces back to 0.7
    result = _elastic_bounce([1.3], 0.0, 1.0)
    np.testing.assert_allclose(result, [0.7], atol=1e-10)

def test_elastic_bounce_single_reflect_lo():
    # -0.3 is 0.3 below lo=0 → bounces to 0.3
    result = _elastic_bounce([-0.3], 0.0, 1.0)
    np.testing.assert_allclose(result, [0.3], atol=1e-10)

def test_elastic_bounce_double_reflect():
    # span=1, value=2.3 → shifted=2.3, modulo(2)=0.3 → in first half → 0.3
    result = _elastic_bounce([2.3], 0.0, 1.0)
    np.testing.assert_allclose(result, [0.3], atol=1e-10)

def test_elastic_bounce_vectorized():
    values = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    result = _elastic_bounce(values, 0.0, 1.0)
    assert result.shape == (5,)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)

def test_elastic_bounce_non_zero_lo():
    # Range [2, 4], span=2. Value 5 → 1 past hi → bounces to 3
    result = _elastic_bounce([5.0], 2.0, 4.0)
    np.testing.assert_allclose(result, [3.0], atol=1e-10)


# ── build_latents ─────────────────────────────────────────────────────────────

def _zero_velocities():
    return np.zeros(N_FACTORS, dtype=np.float32)

def _midpoint_base():
    return np.array([(lo + hi) / 2 for lo, hi in LATENT_RANGES])

def test_build_latents_shape():
    base = _midpoint_base()
    latents = build_latents(base, 16, _zero_velocities())
    assert latents.shape == (16, N_FACTORS)

def test_build_latents_frozen_factor():
    base = _midpoint_base()
    latents = build_latents(base, 8, _zero_velocities())
    for k in range(N_FACTORS):
        np.testing.assert_array_equal(latents[:, k], base[k])

def test_build_latents_noncircular_stays_in_range():
    base = _midpoint_base()
    velocities = _zero_velocities()
    # Activate a non-circular factor (rot_x = index 0) with large velocity
    velocities[0] = 3.0
    lo, hi = LATENT_RANGES[0]
    latents = build_latents(base, 32, velocities)
    assert np.all(latents[:, 0] >= lo - 1e-9)
    assert np.all(latents[:, 0] <= hi + 1e-9)

def test_build_latents_circular_stays_in_range():
    base = _midpoint_base()
    velocities = _zero_velocities()
    # Activate a circular factor (floor_hue = index 3)
    velocities[3] = 2.0
    lo, hi = LATENT_RANGES[3]
    latents = build_latents(base, 32, velocities)
    assert np.all(latents[:, 3] >= lo - 1e-9)
    assert np.all(latents[:, 3] < hi + 1e-9)

def test_build_latents_full_sweep_noncircular():
    base = _midpoint_base()
    velocities = _zero_velocities()
    velocities[0] = 1.0  # rot_x, non-circular, velocity=1 → sweeps full span
    lo, hi = LATENT_RANGES[0]
    latents = build_latents(base, 64, velocities)
    # First frame at base, last frame elastically bounced
    assert latents[0, 0] == pytest.approx(base[0])


def test_build_latents_walk_frozen_factor_stays_fixed():
    # With momentum < 1 and an rng, factors with v0 == 0 must NOT drift.
    base = _midpoint_base()
    velocities = _zero_velocities()
    velocities[0] = 1.0  # only rot_x is active
    rng = np.random.default_rng(0)
    latents = build_latents(base, 32, velocities,
                            velocity_momentum=0.9, rng=rng)
    for k in range(N_FACTORS):
        if k == 0:
            assert not np.allclose(latents[:, k], base[k])   # active factor moves
        else:
            np.testing.assert_array_equal(latents[:, k], base[k])  # frozen stays put


def test_build_latents_walk_momentum_one_matches_linear():
    # momentum == 1.0 must reproduce the constant-velocity path exactly,
    # regardless of whether an rng is passed.
    base = _midpoint_base()
    velocities = _zero_velocities()
    velocities[0] = 0.7
    linear = build_latents(base, 32, velocities)
    walk = build_latents(base, 32, velocities,
                         velocity_momentum=1.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(walk, linear)


# ── sample_velocities ────────────────────────────────────────────────────────

def test_sample_velocities_deterministic():
    v1 = sample_velocities(0.5, 0.0, 'gaussian', 42, 'syn', 'obj', 0)
    v2 = sample_velocities(0.5, 0.0, 'gaussian', 42, 'syn', 'obj', 0)
    np.testing.assert_array_equal(v1, v2)

def test_sample_velocities_different_seqs():
    v0 = sample_velocities(0.5, 0.0, 'gaussian', 42, 'syn', 'obj', 0)
    v1 = sample_velocities(0.5, 0.0, 'gaussian', 42, 'syn', 'obj', 1)
    assert not np.array_equal(v0, v1)

def test_sample_velocities_always_nonzero():
    for seq_idx in range(20):
        v = sample_velocities(0.99, 0.0, 'gaussian', 0, 'syn', 'obj', seq_idx)
        assert np.any(v != 0.0)

def test_sample_velocities_discrete_values():
    # stdev=0 → active factors are exactly ±1.0
    for seq_idx in range(10):
        v = sample_velocities(0.0, 0.0, 'gaussian', 0, 'syn', 'obj', seq_idx)
        active = v[v != 0.0]
        assert np.all(np.abs(active) == 1.0)

def test_sample_velocities_allowed_factors():
    allowed = {0, 1}
    v = sample_velocities(0.0, 0.0, 'gaussian', 0, 'syn', 'obj', 0, allowed_factors=allowed)
    for k in range(N_FACTORS):
        if k not in allowed:
            assert v[k] == 0.0

def test_sample_velocities_uniform_in_range():
    stdev = 1.5
    for seq_idx in range(20):
        v = sample_velocities(0.0, stdev, 'uniform', 0, 'syn', 'obj', seq_idx)
        active = v[v != 0.0]
        assert np.all(np.abs(active) <= stdev + 1e-6)

def test_sample_velocities_gaussian_shape():
    v = sample_velocities(0.5, 1.0, 'gaussian', 0, 'syn', 'obj', 0)
    assert v.shape == (N_FACTORS,)
    assert v.dtype == np.float32


# ── enable_full_rotation ─────────────────────────────────────────────────────
# Mutates module globals in place, so each test snapshots and restores them.

@pytest.fixture
def restore_full_rotation():
    ranges = latent_utils.LATENT_RANGES.copy()
    circular = set(latent_utils.CIRCULAR_FACTORS)
    yield
    latent_utils.LATENT_RANGES[:] = ranges
    latent_utils.CIRCULAR_FACTORS.clear()
    latent_utils.CIRCULAR_FACTORS.update(circular)


def test_enable_full_rotation_widens_all_axes(restore_full_rotation):
    enable_full_rotation()
    for k in ROTATION_FACTORS:
        lo, hi = latent_utils.LATENT_RANGES[k]
        assert lo == pytest.approx(-np.pi)
        assert hi == pytest.approx(np.pi)


def test_enable_full_rotation_marks_axes_circular(restore_full_rotation):
    enable_full_rotation()
    assert set(ROTATION_FACTORS) <= latent_utils.CIRCULAR_FACTORS


def test_enable_full_rotation_mutates_in_place(restore_full_rotation):
    # Callers that imported the names by reference must see the update.
    ranges_ref = latent_utils.LATENT_RANGES
    circular_ref = latent_utils.CIRCULAR_FACTORS
    enable_full_rotation()
    assert latent_utils.LATENT_RANGES is ranges_ref
    assert latent_utils.CIRCULAR_FACTORS is circular_ref
    assert 0 in circular_ref and 1 in circular_ref


def test_enable_full_rotation_makes_rot_x_wrap(restore_full_rotation):
    # Before: rot_x reflects off its boundary. After: it wraps.
    enable_full_rotation()
    lo, hi = latent_utils.LATENT_RANGES[0]
    span = hi - lo
    base = np.array([(l + h) / 2 for l, h in latent_utils.LATENT_RANGES])
    base[0] = hi - 0.05 * span          # start just below the upper edge
    v = np.zeros(N_FACTORS, dtype=np.float32)
    v[0] = 1.0                          # full-span forward sweep, will cross +π
    traj = build_latents(base, 32, v)[:, 0]
    assert (traj >= lo - 1e-9).all() and (traj <= hi + 1e-9).all()
    # Wrapping means the value keeps advancing (mod span) rather than turning
    # back: over a full-span sweep it should visit both halves of the range.
    assert traj.min() < lo + 0.25 * span
    assert traj.max() > hi - 0.25 * span
