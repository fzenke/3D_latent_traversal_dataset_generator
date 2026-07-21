import os
import sys
import pickle
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import TraversalDataset

# Try PIL for writing tiny test frames; skip dataset tests if unavailable.
PIL = pytest.importorskip('PIL', reason='Pillow required for dataset tests')
from PIL import Image


# ── Fixture helpers ───────────────────────────────────────────────────────────

N_FACTORS = 10
LATENT_NAMES = ['rot_x', 'rot_y', 'rot_z', 'floor_hue',
                'spot_theta', 'spot_phi', 'spot_hue',
                'trans_x', 'trans_y', 'trans_z']
LATENT_RANGES = np.zeros((N_FACTORS, 2), dtype=np.float32)
T = 8
IMG_SIZE = 4


def _write_frames(seq_dir, n_frames, img_size):
    os.makedirs(seq_dir, exist_ok=True)
    img = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    for t in range(n_frames):
        img.save(os.path.join(seq_dir, f'frame_{t:04d}.jpg'))


def _make_sequence(root, synset, obj_id, seq_idx, n_frames, img_size):
    frames_dir = os.path.join('seqs', synset, obj_id[:2], obj_id, f'seq_{seq_idx:04d}')
    seq_dir = os.path.join(root, frames_dir)
    _write_frames(seq_dir, n_frames, img_size)
    return {
        'synset_id': synset,
        'obj_id': obj_id,
        'seq_idx': seq_idx,
        'traversal_factors': [seq_idx % N_FACTORS],
        'traversal_velocities': np.zeros(N_FACTORS, dtype=np.float32),
        'base_latent': np.zeros(N_FACTORS, dtype=np.float32),
        'latents': np.zeros((n_frames, N_FACTORS), dtype=np.float32),
        'frames_dir': frames_dir,
    }


def _make_dataset(tmp_path, sequences_spec):
    """
    sequences_spec: list of (synset, obj_id, seq_idx) tuples.
    Returns (pkl_path, root_dir).
    """
    root = str(tmp_path / 'dataset')
    sequences = [
        _make_sequence(root, syn, obj, idx, T, IMG_SIZE)
        for syn, obj, idx in sequences_spec
    ]
    meta = {
        'latent_names': LATENT_NAMES,
        'latent_ranges': LATENT_RANGES,
        'n_frames': T,
        'image_size': IMG_SIZE,
        'sequences': sequences,
    }
    pkl_path = os.path.join(root, 'metadata.pkl')
    os.makedirs(root, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(meta, f)
    return pkl_path, root


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def ds(tmp_path):
    spec = [
        ('02933112', 'obj_aaa', 0),
        ('02933112', 'obj_aaa', 1),
        ('02691156', 'obj_xxx', 0),
    ]
    pkl, root = _make_dataset(tmp_path, spec)
    return TraversalDataset(pkl, root)


def test_len(ds):
    assert len(ds) == 3

def test_getitem_keys(ds):
    item = ds[0]
    for key in ('frames', 'latents', 'base_latent', 'synset_id', 'obj_id',
                 'traversal_factors', 'traversal_velocities'):
        assert key in item

def test_getitem_frames_shape(ds):
    import torch
    item = ds[0]
    assert item['frames'].shape == (T, 3, IMG_SIZE, IMG_SIZE)

def test_getitem_latents_shape(ds):
    item = ds[0]
    assert item['latents'].shape == (T, N_FACTORS)

def test_getitem_base_latent_shape(ds):
    item = ds[0]
    assert item['base_latent'].shape == (N_FACTORS,)

def test_getitem_velocities_shape(ds):
    item = ds[0]
    assert item['traversal_velocities'].shape == (N_FACTORS,)

def test_filter_by_factor_int(ds):
    subset = ds.filter_by_factor(0)
    # Two sequences (obj_aaa seq 0 and obj_xxx seq 0) have traversal_factor 0
    assert len(subset) == 2

def test_filter_by_factor_str(ds):
    subset = ds.filter_by_factor('rot_x')  # index 0
    assert len(subset) == 2

def test_filter_by_synset(ds):
    subset = ds.filter_by_synset('02933112')
    assert len(subset) == 2
    for i in range(len(subset)):
        assert subset[i]['synset_id'] == '02933112'

def test_subset_len(ds):
    subset = ds.filter_by_synset('02691156')
    assert len(subset) == 1

def test_subset_getitem(ds):
    subset = ds.filter_by_synset('02691156')
    item = subset[0]
    assert item['synset_id'] == '02691156'

def test_latent_names_exposed(ds):
    assert ds.latent_names == LATENT_NAMES

def test_latent_ranges_shape(ds):
    assert ds.latent_ranges.shape == (N_FACTORS, 2)
