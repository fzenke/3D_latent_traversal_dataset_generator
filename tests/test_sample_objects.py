import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sample_objects import collect_objects, load_exclusions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_shapenet(tmp_path, layout):
    """Build a minimal fake ShapeNet tree.

    layout: dict mapping synset_id -> list of obj_ids to create.
    Returns the path to the root.
    """
    root = tmp_path / 'ShapeNet'
    for synset_id, obj_ids in layout.items():
        for obj_id in obj_ids:
            model_dir = root / synset_id / obj_id / 'models'
            model_dir.mkdir(parents=True)
            (model_dir / 'model_normalized.obj').write_text('')
    return str(root)


SYNSET_A = '02933112'
SYNSET_B = '02691156'
OBJ_IDS_A = ['obj_aaa', 'obj_bbb', 'obj_ccc', 'obj_ddd']
OBJ_IDS_B = ['obj_xxx', 'obj_yyy']


@pytest.fixture
def shapenet(tmp_path):
    return _make_shapenet(tmp_path, {SYNSET_A: OBJ_IDS_A, SYNSET_B: OBJ_IDS_B})


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_random_sample_count(shapenet):
    pairs = collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=0)
    assert len(pairs) == 2
    assert all(s == SYNSET_A for s, _ in pairs)

def test_random_sample_all(shapenet):
    pairs = collect_objects(shapenet, synsets=[(SYNSET_A, len(OBJ_IDS_A))], seed=0)
    assert len(pairs) == len(OBJ_IDS_A)

def test_random_sample_deterministic(shapenet):
    p1 = collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=42)
    p2 = collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=42)
    assert p1 == p2

def test_random_sample_different_seeds(shapenet):
    # With 4 objects and sample size 2, different seeds can differ
    results = {tuple(sorted(collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=s)))
               for s in range(20)}
    assert len(results) > 1

def test_pinned_object_included(shapenet):
    pairs = collect_objects(shapenet, objects=[(SYNSET_A, 'obj_aaa')], seed=0)
    assert (SYNSET_A, 'obj_aaa') in pairs

def test_deduplication(shapenet):
    # If a pinned object is also drawn in random sample, it should appear only once
    pairs = collect_objects(
        shapenet,
        synsets=[(SYNSET_A, len(OBJ_IDS_A))],  # draws all, including obj_aaa
        objects=[(SYNSET_A, 'obj_aaa')],
        seed=0,
    )
    obj_ids = [o for _, o in pairs]
    assert obj_ids.count('obj_aaa') == 1

def test_pinned_object_preserved_when_also_sampled(shapenet):
    pairs = collect_objects(
        shapenet,
        synsets=[(SYNSET_A, len(OBJ_IDS_A))],
        objects=[(SYNSET_A, 'obj_aaa')],
        seed=0,
    )
    assert (SYNSET_A, 'obj_aaa') in pairs

def test_multi_synset(shapenet):
    pairs = collect_objects(
        shapenet,
        synsets=[(SYNSET_A, 2), (SYNSET_B, 1)],
        seed=0,
    )
    synsets_seen = {s for s, _ in pairs}
    assert SYNSET_A in synsets_seen
    assert SYNSET_B in synsets_seen
    assert len(pairs) == 3

def test_missing_synset_errors(shapenet):
    with pytest.raises(SystemExit):
        collect_objects(shapenet, synsets=[('99999999', 1)], seed=0)

def test_missing_object_errors(shapenet):
    with pytest.raises(SystemExit):
        collect_objects(shapenet, objects=[(SYNSET_A, 'no_such_obj')], seed=0)

def test_n_exceeds_available_errors(shapenet):
    with pytest.raises(SystemExit):
        collect_objects(shapenet, synsets=[(SYNSET_A, len(OBJ_IDS_A) + 1)], seed=0)

def test_all_objects_from_synset(shapenet):
    pairs = collect_objects(shapenet, synsets=[(SYNSET_A, None)], seed=0)
    assert len(pairs) == len(OBJ_IDS_A)
    assert {o for _, o in pairs} == set(OBJ_IDS_A)

def test_returns_list_of_tuples(shapenet):
    pairs = collect_objects(shapenet, synsets=[(SYNSET_A, 1)], seed=0)
    assert isinstance(pairs, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


# ── exclusions ────────────────────────────────────────────────────────────────

def test_exclude_removes_object(shapenet):
    excl = {f"{SYNSET_A}/obj_aaa"}
    pairs = collect_objects(shapenet, synsets=[(SYNSET_A, None)], seed=0, exclude=excl)
    assert (SYNSET_A, 'obj_aaa') not in pairs
    assert len(pairs) == len(OBJ_IDS_A) - 1

def test_exclude_draws_replacement_keeping_count(shapenet):
    # 4 objects, want 2. Excluding one must still yield 2 from the other three,
    # never a gap — this is the whole point of excluding before sampling.
    for seed in range(15):
        excl = {f"{SYNSET_A}/obj_aaa"}
        pairs = collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=seed, exclude=excl)
        assert len(pairs) == 2
        assert (SYNSET_A, 'obj_aaa') not in pairs

def test_exclude_insufficient_pool_errors(shapenet):
    # 4 objects, exclude 3, request 2 -> only 1 valid -> must error.
    excl = {f"{SYNSET_A}/{o}" for o in OBJ_IDS_A[:3]}
    with pytest.raises(SystemExit):
        collect_objects(shapenet, synsets=[(SYNSET_A, 2)], seed=0, exclude=excl)

def test_exclude_drops_pinned_object(shapenet):
    excl = {f"{SYNSET_A}/obj_aaa"}
    pairs = collect_objects(shapenet, objects=[(SYNSET_A, 'obj_aaa')], seed=0, exclude=excl)
    assert (SYNSET_A, 'obj_aaa') not in pairs

def test_load_exclusions(tmp_path):
    p = tmp_path / 'bad.txt'
    p.write_text("# comment\n02933112/obj_aaa\n\n02691156/obj_xxx\n")
    excl = load_exclusions([str(p)])
    assert excl == {'02933112/obj_aaa', '02691156/obj_xxx'}

def test_load_exclusions_missing_file_errors():
    with pytest.raises(SystemExit):
        load_exclusions(['/no/such/file.txt'])
