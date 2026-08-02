"""Tests for the train/val split (ml/train.py).

Guards a bug that silently inflated every held-out metric: CatFLW stems are
`<subject>_<shot>` over ~339 cats with a median of 6 photos each, so splitting on
image index put another photo of the same cat in training for 310 of 311 val
images. Nothing failed -- val loss just measured memorization. See
docs/heatmap_decoding.md.

torch is an ml/-only dependency, so these skip cleanly without it.
"""
import pytest

pytest.importorskip("torch")

from ml.dataset import subject_of  # noqa: E402
from ml.train import _subject_disjoint_split  # noqa: E402


def _stems(subjects=40, shots=6):
    return [f"{s:08d}_{shot:03d}" for s in range(1, subjects + 1) for shot in range(shots)]


def test_subject_of_strips_shot_number():
    assert subject_of("00000001_012") == "00000001"
    assert subject_of("CAT_01_00000107_006") == "CAT_01_00000107"


def test_subject_of_leaves_unnumbered_stem_alone():
    assert subject_of("plain_name") == "plain_name"


def test_split_never_shares_a_cat_across_train_and_val():
    stems = _stems()
    train, val = _subject_disjoint_split(stems, val_split=0.15, seed=42)

    train_subjects = {subject_of(stems[i]) for i in train}
    val_subjects = {subject_of(stems[i]) for i in val}
    assert not (train_subjects & val_subjects)


def test_split_covers_every_sample_exactly_once():
    stems = _stems()
    train, val = _subject_disjoint_split(stems, val_split=0.15, seed=42)

    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(len(stems)))


def test_split_lands_near_the_requested_size():
    stems = _stems()
    _, val = _subject_disjoint_split(stems, val_split=0.15, seed=42)
    # Whole cats move together, so the val fraction overshoots by at most one cat.
    assert 0.15 <= len(val) / len(stems) <= 0.15 + 6 / len(stems)


def test_split_is_deterministic_for_a_seed():
    stems = _stems()
    assert _subject_disjoint_split(stems, 0.15, 42) == _subject_disjoint_split(stems, 0.15, 42)


def test_different_seeds_choose_different_cats():
    stems = _stems()
    _, val_a = _subject_disjoint_split(stems, 0.15, 1)
    _, val_b = _subject_disjoint_split(stems, 0.15, 2)
    assert val_a != val_b


def test_single_shot_subjects_still_split_cleanly():
    stems = _stems(subjects=50, shots=1)
    train, val = _subject_disjoint_split(stems, val_split=0.2, seed=7)
    assert not {subject_of(stems[i]) for i in train} & {subject_of(stems[i]) for i in val}
    assert len(val) >= 1
