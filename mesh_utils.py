"""Mesh cleanup helpers that need no bpy, so they can be unit-tested.

The geometry defect these target: ShapeNet meshes routinely carry duplicate
faces built on the SAME vertex indices. A cup measured here had 270 vertices
supporting 968 faces, where Euler's formula caps a genus-1 triangle surface at
about 540 — nearly twice the faces the vertex set can legitimately support.

Because the duplicates share vertices rather than merely sitting at the same
coordinates, `remove_doubles` finds nothing to weld ("Removed 0 vertices") and
Solidify offsets both copies together. The overlap has to be resolved at the
face level instead.
"""


def duplicate_face_indices(faces):
    """Indices of faces that repeat an earlier face's vertex set.

    `faces` is an iterable of vertex-index sequences, one per face. Returns the
    positions of every face whose vertex set was already seen, keeping the first
    occurrence of each. The caller deletes those faces.

    The key is order-independent (sorted), so a face duplicated with reversed
    winding — the common double-sided authoring trick, and a case Solidify
    actively cannot fix — is still recognised as a duplicate.

    Degenerate faces that name the same vertex more than once collapse to the
    same key as their deduplicated form, which is intended: they enclose no area
    and are safe to drop as duplicates.
    """
    seen = set()
    dupes = []
    for i, verts in enumerate(faces):
        key = tuple(sorted(set(verts)))
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


def euler_face_budget(n_verts, genus=0):
    """Max faces a closed triangle surface of `n_verts` can have: 2V - 4 + 4g.

    Used only for diagnostics. Exceeding it proves the mesh cannot be a clean
    manifold surface — it has duplicate or otherwise overlapping faces.
    """
    return 2 * n_verts - 4 + 4 * genus
