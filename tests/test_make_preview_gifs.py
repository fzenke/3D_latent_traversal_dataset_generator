import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PIL = pytest.importorskip('PIL', reason='Pillow required for gif tests')
from PIL import Image  # noqa: E402

from make_preview_gifs import (  # noqa: E402
    find_sequence_dirs, first_sequence_per_object, frame_paths, gif_name,
    write_gif, _seq_index,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_frames(seq_dir, n, size=8, start=0):
    os.makedirs(seq_dir, exist_ok=True)
    for i in range(start, start + n):
        img = Image.new('RGB', (size, size), (i * 7 % 256, 100, 150))
        img.save(os.path.join(seq_dir, f'frame_{i:04d}.jpg'))


def _make_dataset(root, layout):
    """layout: {(synset, obj_id): [seq_idx, ...]} -> canonical seqs/ tree."""
    for (synset, obj_id), seq_ids in layout.items():
        for s in seq_ids:
            seq_dir = os.path.join(root, 'seqs', synset, obj_id[:2], obj_id,
                                   f'seq_{s:04d}')
            _make_frames(seq_dir, 3)
    return root


# ── discovery ────────────────────────────────────────────────────────────────

def test_seq_index_parses():
    assert _seq_index('/a/b/seq_0007') == 7
    assert _seq_index('/a/b/notaseq') is None


def test_finds_sequences_in_canonical_layout(tmp_path):
    root = _make_dataset(str(tmp_path), {('0300', 'abcdef'): [0, 1]})
    found = find_sequence_dirs(root)
    assert len(found) == 2


def test_falls_back_to_recursive_walk(tmp_path):
    # No seqs/ prefix — a bare directory of seq_* dirs (like frames/).
    _make_frames(str(tmp_path / 'seq_0000'), 3)
    found = find_sequence_dirs(str(tmp_path))
    assert len(found) == 1
    assert found[0].endswith('seq_0000')


def test_first_sequence_picks_lowest_index(tmp_path):
    root = _make_dataset(str(tmp_path), {('0300', 'abcdef'): [5, 2, 9]})
    firsts = first_sequence_per_object(find_sequence_dirs(root))
    assert len(firsts) == 1
    assert list(firsts.values())[0].endswith('seq_0002')


def test_first_sequence_lowest_is_not_zero(tmp_path):
    # Single-factor mode: seq_idx IS the factor index, so 0000 may not exist.
    root = _make_dataset(str(tmp_path), {('0300', 'abcdef'): [3, 7]})
    firsts = first_sequence_per_object(find_sequence_dirs(root))
    assert list(firsts.values())[0].endswith('seq_0003')


def test_one_entry_per_object(tmp_path):
    root = _make_dataset(str(tmp_path), {
        ('0300', 'aaaaaa'): [0, 1],
        ('0300', 'bbbbbb'): [0],
        ('0400', 'cccccc'): [2, 4],
    })
    firsts = first_sequence_per_object(find_sequence_dirs(root))
    assert len(firsts) == 3


# ── frame ordering ───────────────────────────────────────────────────────────

def test_frame_paths_numeric_order(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    _make_frames(seq, 12)
    got = [os.path.basename(p) for p in frame_paths(seq)]
    assert got == [f'frame_{i:04d}.jpg' for i in range(12)]


def test_frame_paths_ignores_non_frames(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    _make_frames(seq, 2)
    open(os.path.join(seq, 'notes.txt'), 'w').close()
    assert len(frame_paths(seq)) == 2


def test_frame_paths_empty_dir(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    os.makedirs(seq)
    assert frame_paths(seq) == []


# ── naming ───────────────────────────────────────────────────────────────────

def test_gif_name_from_canonical_layout():
    assert gif_name('/d/seqs/03928116/13/13394ca') == '03928116_13394ca.gif'


def test_gif_name_fallback_without_prefix_dir():
    assert gif_name('/some/where/myobject') == 'myobject.gif'


# ── encoding ─────────────────────────────────────────────────────────────────

def test_write_gif_frame_count(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    _make_frames(seq, 6)
    out = str(tmp_path / 'out' / 'x.gif')
    n = write_gif(frame_paths(seq), out, fps=20)
    assert n == 6
    assert os.path.exists(out)
    with Image.open(out) as im:
        assert im.is_animated
        assert im.n_frames == 6


def test_write_gif_respects_resize(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    _make_frames(seq, 2, size=8)
    out = str(tmp_path / 'y.gif')
    write_gif(frame_paths(seq), out, resize=32)
    with Image.open(out) as im:
        assert im.size == (32, 32)


def test_write_gif_duration_from_fps(tmp_path):
    seq = str(tmp_path / 'seq_0000')
    _make_frames(seq, 3)
    out = str(tmp_path / 'z.gif')
    write_gif(frame_paths(seq), out, fps=10)  # -> 100 ms
    with Image.open(out) as im:
        assert im.info['duration'] == 100


def test_write_gif_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        write_gif([], str(tmp_path / 'empty.gif'))


def test_frames_are_opaque_and_not_disposed_to_background(tmp_path):
    """Regression: disposal=2 makes viewers clear to transparent between frames.

    That renders as a grey/white checkerboard flashing over the animation, since
    'restore to background' is implemented as transparent. Every frame here is
    fully opaque, so disposal must be 1 ('do not dispose') and no frame may
    contain transparent pixels.
    """
    seq = tmp_path / 'seq_0000'
    seq.mkdir()
    for i in range(5):
        Image.new('RGB', (16, 16), (200, 60 + i * 20, 90)).save(
            seq / f'frame_{i:04d}.jpg')

    out = str(tmp_path / 'opaque.gif')
    write_gif(frame_paths(str(seq)), out, fps=10)

    with Image.open(out) as im:
        for i in range(im.n_frames):
            im.seek(i)
            assert im.disposal_method != 2, f"frame {i} disposes to background"
            alpha_min, _ = im.convert('RGBA').getchannel('A').getextrema()
            assert alpha_min == 255, f"frame {i} has transparent pixels"


def test_palette_covers_hues_absent_from_first_frame(tmp_path):
    """Regression: the palette must come from ALL frames, not frame 0.

    These traversals sweep floor_hue/spot_hue, so late frames contain colours
    absent from frame 0. A frame-0-only palette mapped them wrong and collapsed
    the frames into one merged image.
    """
    seq = tmp_path / 'seq_0000'
    seq.mkdir()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # no shared hues
    for i, c in enumerate(colors):
        Image.new('RGB', (8, 8), c).save(seq / f'frame_{i:04d}.jpg')

    out = str(tmp_path / 'hues.gif')
    write_gif(frame_paths(str(seq)), out, fps=10)

    with Image.open(out) as im:
        assert im.is_animated
        assert im.n_frames == 3
        seen = []
        for i in range(im.n_frames):
            im.seek(i)
            seen.append(im.convert('RGB').getpixel((4, 4)))

    # Each frame must stay near its own hue — i.e. the dominant channel wins.
    for got, want in zip(seen, colors):
        assert got.index(max(got)) == want.index(max(want)), f"{got} vs {want}"
