import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mesh_utils import duplicate_face_indices, euler_face_budget


def test_no_duplicates():
    assert duplicate_face_indices([[0, 1, 2], [1, 2, 3], [2, 3, 4]]) == []


def test_exact_duplicate_is_flagged():
    assert duplicate_face_indices([[0, 1, 2], [0, 1, 2]]) == [1]


def test_first_occurrence_is_kept():
    """Index 0 must survive — deleting both copies would punch a hole."""
    dupes = duplicate_face_indices([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    assert dupes == [1, 2]
    assert 0 not in dupes


def test_reversed_winding_counts_as_duplicate():
    """Double-sided authoring: same face, flipped normal. Solidify can't fix it."""
    assert duplicate_face_indices([[0, 1, 2], [2, 1, 0]]) == [1]


def test_rotated_vertex_order_counts_as_duplicate():
    assert duplicate_face_indices([[0, 1, 2, 3], [2, 3, 0, 1]]) == [1]


def test_different_faces_sharing_vertices_are_kept():
    """Adjacent faces share an edge; they are not duplicates."""
    assert duplicate_face_indices([[0, 1, 2], [0, 1, 3], [0, 2, 3]]) == []


def test_quads_and_tris_mixed():
    faces = [[0, 1, 2], [0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2]]
    assert duplicate_face_indices(faces) == [2, 3]


def test_empty_mesh():
    assert duplicate_face_indices([]) == []


def test_degenerate_face_matches_its_collapsed_form():
    # [0,1,1] encloses no area; it keys the same as the edge-like set {0,1}.
    assert duplicate_face_indices([[0, 1], [0, 1, 1]]) == [1]


def test_euler_budget_sphere_and_genus_one():
    assert euler_face_budget(270) == 536
    assert euler_face_budget(270, genus=1) == 540


def test_measured_cup_exceeds_budget():
    """The observed cup: 270 verts / 968 faces cannot be a clean surface."""
    assert 968 > euler_face_budget(270, genus=1)
