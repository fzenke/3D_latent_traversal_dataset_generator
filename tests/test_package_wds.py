import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from package_wds import n_objects, objects_by_synset, split_sequences


def make_sequences(n_synsets=3, n_objects_per_synset=10, n_seqs=20):
    """Synthetic metadata: only the fields the splitter looks at."""
    seqs = []
    for s in range(n_synsets):
        for o in range(n_objects_per_synset):
            for i in range(n_seqs):
                seqs.append({'synset_id': f'0{s}000000',
                             'obj_id': f'obj{o:03d}',
                             'seq_idx': i})
    return seqs


def objects_of(seqs):
    return {(s['synset_id'], s['obj_id']) for s in seqs}


# ── sequence-level split ─────────────────────────────────────────────────────

def test_sequence_split_sizes():
    seqs = make_sequences()          # 3 * 10 * 20 = 600
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='sequence')
    assert len(parts['train']) == 480
    assert len(parts['val']) == 60
    assert len(parts['test']) == 60


def test_sequence_split_is_a_partition():
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='sequence')
    ids = [id(s) for part in parts.values() for s in part]
    assert len(ids) == len(seqs)
    assert len(set(ids)) == len(seqs)   # no sequence duplicated or dropped


def test_sequence_split_shares_objects():
    """The documented limitation: objects are NOT held out in this mode."""
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='sequence')
    assert objects_of(parts['train']) & objects_of(parts['test'])


# ── object-level split ───────────────────────────────────────────────────────

def test_object_split_holds_out_whole_objects():
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    train, val, test = (objects_of(parts[k]) for k in ('train', 'val', 'test'))
    assert not train & val
    assert not train & test
    assert not val & test


def test_object_split_is_a_partition():
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    ids = [id(s) for part in parts.values() for s in part]
    assert len(ids) == len(seqs)
    assert len(set(ids)) == len(seqs)


def test_object_split_is_stratified_by_synset():
    """Every category must appear in every split, not just overall."""
    seqs = make_sequences(n_synsets=3, n_objects_per_synset=10)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    for name in ('train', 'val', 'test'):
        synsets = {s['synset_id'] for s in parts[name]}
        assert len(synsets) == 3, f"{name} is missing categories: {synsets}"


def test_object_split_respects_fractions():
    seqs = make_sequences(n_synsets=3, n_objects_per_synset=10)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    # 10 objects per synset, 10% each -> 1 val + 1 test + 8 train per synset
    assert n_objects(parts['train']) == 24
    assert n_objects(parts['val']) == 3
    assert n_objects(parts['test']) == 3


def test_object_split_keeps_all_sequences_of_an_object_together():
    seqs = make_sequences(n_synsets=2, n_objects_per_synset=10, n_seqs=7)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    for part in parts.values():
        counts = {}
        for s in part:
            counts[(s['synset_id'], s['obj_id'])] = \
                counts.get((s['synset_id'], s['obj_id']), 0) + 1
        assert all(c == 7 for c in counts.values())


def test_object_split_is_deterministic_given_seed():
    seqs = make_sequences()
    a = split_sequences(seqs, 0.1, 0.1, seed=3, split_by='object')
    b = split_sequences(seqs, 0.1, 0.1, seed=3, split_by='object')
    assert objects_of(a['test']) == objects_of(b['test'])


def test_object_split_seed_changes_the_partition():
    seqs = make_sequences()
    a = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    b = split_sequences(seqs, 0.1, 0.1, seed=1, split_by='object')
    assert objects_of(a['test']) != objects_of(b['test'])


def test_object_split_never_starves_training():
    """A synset with 2 objects must still contribute to train."""
    warnings = []
    seqs = make_sequences(n_synsets=1, n_objects_per_synset=2, n_seqs=5)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object',
                            warn=warnings.append)
    assert n_objects(parts['train']) >= 1
    assert warnings, "a category too small to populate every split must warn"


def test_object_split_single_object_synset_goes_to_train():
    seqs = make_sequences(n_synsets=1, n_objects_per_synset=1, n_seqs=4)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object',
                            warn=lambda _m: None)
    assert len(parts['train']) == 4
    assert 'val' not in parts and 'test' not in parts


def test_object_split_shuffles_within_split():
    """Shards must not end up grouped by object."""
    seqs = make_sequences(n_synsets=2, n_objects_per_synset=10, n_seqs=20)
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    train = parts['train']
    keys = [(s['synset_id'], s['obj_id']) for s in train]
    runs = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    # If grouped by object there would be ~n_objects-1 transitions; shuffled
    # order gives a transition at nearly every step.
    assert runs > 0.8 * len(keys)


def test_no_holdout_returns_train_only():
    seqs = make_sequences(n_synsets=1, n_objects_per_synset=4, n_seqs=3)
    parts = split_sequences(seqs, 0.0, 0.0, seed=0, split_by='object')
    assert set(parts) == {'train'}
    assert len(parts['train']) == 12


def test_n_objects_counts_unique_pairs():
    seqs = make_sequences(n_synsets=2, n_objects_per_synset=5, n_seqs=3)
    assert n_objects(seqs) == 10


# ── recorded split membership ────────────────────────────────────────────────

def test_objects_by_synset_groups_and_sorts():
    seqs = make_sequences(n_synsets=2, n_objects_per_synset=3, n_seqs=2)
    got = objects_by_synset(seqs)
    assert set(got) == {'00000000', '01000000'}
    assert got['00000000'] == ['obj000', 'obj001', 'obj002']


def test_objects_by_synset_deduplicates():
    seqs = make_sequences(n_synsets=1, n_objects_per_synset=2, n_seqs=50)
    assert objects_by_synset(seqs) == {'00000000': ['obj000', 'obj001']}


def test_recorded_membership_matches_the_split():
    """dataset_info.json's 'objects' must name exactly the split's objects."""
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    for name, part in parts.items():
        recorded = {(sid, oid)
                    for sid, oids in objects_by_synset(part).items()
                    for oid in oids}
        assert recorded == objects_of(part)


def test_recorded_membership_is_disjoint_for_object_split():
    seqs = make_sequences()
    parts = split_sequences(seqs, 0.1, 0.1, seed=0, split_by='object')
    listed = {}
    for name, part in parts.items():
        listed[name] = {(sid, oid)
                        for sid, oids in objects_by_synset(part).items()
                        for oid in oids}
    assert not listed['train'] & listed['val']
    assert not listed['train'] & listed['test']
    assert not listed['val'] & listed['test']
