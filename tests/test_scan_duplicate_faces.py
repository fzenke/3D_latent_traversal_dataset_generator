import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scan_duplicate_faces import parse_obj_faces, scan_model


def _write_obj(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(text)
    return path


CLEAN = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
f 2 4 3
"""

# Same two faces, then both repeated — the wholly-doubled pattern seen on the cup.
DOUBLED = CLEAN + """\
f 1 2 3
f 2 4 3
"""


def test_parses_verts_and_faces(tmp_path):
    p = _write_obj(str(tmp_path / 'a.obj'), CLEAN)
    n_verts, faces = parse_obj_faces(p)
    assert n_verts == 4
    assert faces == [[1, 2, 3], [2, 4, 3]]


def test_parses_slash_forms(tmp_path):
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"
    p = _write_obj(str(tmp_path / 'b.obj'), text)
    _n, faces = parse_obj_faces(p)
    assert faces == [[1, 2, 3]]


def test_parses_double_slash_form(tmp_path):
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//1 2//2 3//3\n"
    p = _write_obj(str(tmp_path / 'c.obj'), text)
    _n, faces = parse_obj_faces(p)
    assert faces == [[1, 2, 3]]


def test_negative_indices_are_relative(tmp_path):
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    p = _write_obj(str(tmp_path / 'd.obj'), text)
    _n, faces = parse_obj_faces(p)
    assert faces == [[1, 2, 3]]


def test_ignores_normals_and_uvs_as_vertices(tmp_path):
    text = "v 0 0 0\nvn 0 0 1\nvt 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    p = _write_obj(str(tmp_path / 'e.obj'), text)
    n_verts, _faces = parse_obj_faces(p)
    assert n_verts == 3  # vn / vt must not be counted


def test_scan_clean_model(tmp_path):
    _write_obj(str(tmp_path / '0300' / 'obj1' / 'models' /
                   'model_normalized.obj'), CLEAN)
    row = scan_model(str(tmp_path), '0300', 'obj1')
    assert row['n_faces'] == 2
    assert row['n_dupes'] == 0
    assert row['dupe_frac'] == 0.0


def test_scan_doubled_model_scores_half(tmp_path):
    """A wholly doubled mesh must score exactly 0.5, like the real cup."""
    _write_obj(str(tmp_path / '0300' / 'obj2' / 'models' /
                   'model_normalized.obj'), DOUBLED)
    row = scan_model(str(tmp_path), '0300', 'obj2')
    assert row['n_faces'] == 4
    assert row['n_dupes'] == 2
    assert row['dupe_frac'] == 0.5


def test_scan_missing_obj_returns_none(tmp_path):
    assert scan_model(str(tmp_path), '0300', 'nope') is None
