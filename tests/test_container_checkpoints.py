"""Tests for local-checkpoint resolution in the composition root.

build_service used to require GG_LOCAL_CHECKPOINT to be set; it now defaults to the
subject-disjoint-split checkpoints, so the failure mode moved from "env var unset" to
"file missing". These pin the default *targets* without needing the real checkpoints,
which are gitignored (ml/checkpoints/).
"""
import pytest

from grimaceguide import container


def test_defaults_point_at_the_subject_disjoint_checkpoints():
    """The leaky-split checkpoints must never become the default -- see docs/heatmap_decoding.md."""
    assert container.DEFAULT_LOCAL_CHECKPOINT.name == "landmark_net_subjsplit.pt"
    assert container.DEFAULT_LOCAL_BBOX_CHECKPOINT.name == "bbox_net_subjsplit.pt"


def test_missing_landmark_checkpoint_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GG_LOCAL_CHECKPOINT", str(tmp_path / "nope.pt"))
    with pytest.raises(RuntimeError) as excinfo:
        container.build_service(landmark_source="local")
    message = str(excinfo.value)
    assert "not found" in message
    assert "nope.pt" in message  # names the path it actually looked at
    assert "ml/README.md" in message  # points at how to fix it


def test_unknown_landmark_source_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown GG_LANDMARK_SOURCE"):
        container.build_service(landmark_source="banana")
