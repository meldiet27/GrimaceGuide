"""Tests for heatmap decoding (ml/model.py).

decode_heatmaps sits between the trained model and every geometric formula in
core/scoring.py, so a silent regression here degrades landmark precision without
failing anything else. See docs/heatmap_decoding.md for why sub-pixel refinement
matters: the AUs measuring short distances (muzzle, eyes, whiskers) have a
noise-to-signal ratio around 20%, so tenths of a heatmap cell are not negligible.

torch is an ml/-only dependency (ml/requirements.txt), not needed by the app's
core, so these skip cleanly when it isn't installed.
"""
import pytest

torch = pytest.importorskip("torch")

from ml.model import decode_heatmaps  # noqa: E402  (must follow importorskip)

SIZE = 64


def _gaussian_heatmap(cx: float, cy: float, sigma: float = 1.5, size: int = SIZE):
    """Renders a single-landmark heatmap batch peaked at (cx, cy) in cell coordinates."""
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    heatmap = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return heatmap.view(1, 1, size, size)


def _decoded_cell(heatmaps, mode):
    """Decodes to normalized coords, then back to cell coordinates for comparison."""
    out = decode_heatmaps(heatmaps, mode=mode)[0]
    return float(out[0]) * (SIZE - 1), float(out[1]) * (SIZE - 1)


def test_argmax_recovers_exact_grid_peak():
    # Tolerance covers the float32 round-trip through normalized coordinates
    # (divide by SIZE-1 to normalize, multiply back to compare), not slack in the decode.
    x, y = _decoded_cell(_gaussian_heatmap(20.0, 33.0), "argmax")
    assert x == pytest.approx(20.0, abs=1e-4)
    assert y == pytest.approx(33.0, abs=1e-4)


def test_argmax_quantizes_subpixel_peak_to_grid():
    """The limitation that motivated refinement: argmax can only return integers."""
    x, y = _decoded_cell(_gaussian_heatmap(20.4, 33.4), "argmax")
    assert x == pytest.approx(20.0, abs=1e-4)
    assert y == pytest.approx(33.0, abs=1e-4)


@pytest.mark.parametrize("mode", ["quarter", "parabolic", "soft"])
def test_refinement_beats_argmax_on_subpixel_peak(mode):
    true_x, true_y = 20.35, 33.35
    heatmaps = _gaussian_heatmap(true_x, true_y)

    ax, ay = _decoded_cell(heatmaps, "argmax")
    rx, ry = _decoded_cell(heatmaps, mode)

    argmax_error = abs(ax - true_x) + abs(ay - true_y)
    refined_error = abs(rx - true_x) + abs(ry - true_y)
    assert refined_error < argmax_error


def test_parabolic_is_near_exact_on_a_gaussian():
    """A parabola fits log-Gaussian curvature closely, so error should be well under a cell."""
    true_x, true_y = 20.3, 33.7
    x, y = _decoded_cell(_gaussian_heatmap(true_x, true_y), "parabolic")
    assert abs(x - true_x) < 0.2
    assert abs(y - true_y) < 0.2


def test_refined_offset_never_exceeds_half_a_cell():
    """Refinement nudges within the winning cell; it must not jump to a neighbour."""
    for cx, cy in [(10.5, 10.5), (41.9, 7.1), (0.4, 63.6)]:
        heatmaps = _gaussian_heatmap(cx, cy)
        ax, ay = _decoded_cell(heatmaps, "argmax")
        px, py = _decoded_cell(heatmaps, "parabolic")
        assert abs(px - ax) <= 0.5 + 1e-6
        assert abs(py - ay) <= 0.5 + 1e-6


def test_peak_on_border_stays_in_bounds():
    """Border peaks have no neighbour on one side -- must not extrapolate off the grid."""
    for cx, cy in [(0.0, 30.0), (SIZE - 1.0, 30.0), (30.0, 0.0), (30.0, SIZE - 1.0)]:
        for mode in ("quarter", "parabolic", "soft"):
            x, y = _decoded_cell(_gaussian_heatmap(cx, cy), mode)
            assert 0.0 <= x <= SIZE - 1
            assert 0.0 <= y <= SIZE - 1


def test_flat_heatmap_does_not_produce_nan():
    """A degenerate all-equal heatmap has no concave peak; decoding must stay finite."""
    flat = torch.zeros(1, 1, SIZE, SIZE)
    for mode in ("argmax", "quarter", "parabolic", "soft"):
        out = decode_heatmaps(flat, mode=mode)
        assert torch.isfinite(out).all(), mode


def test_output_shape_and_range_for_multiple_landmarks():
    heatmaps = torch.cat([_gaussian_heatmap(10.0, 20.0), _gaussian_heatmap(50.5, 40.5)], dim=1)
    out = decode_heatmaps(heatmaps)
    assert out.shape == (1, 4)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        decode_heatmaps(_gaussian_heatmap(20.0, 20.0), mode="nope")
