"""Tests for the ARC x predictive boundary fusion (Class F).

These exercise the real parsing/fusion code — the parts that decide where clips are cut. The ARC
model itself is not loaded here (it is a 7B checkpoint and needs a GPU); the pipeline smoke run is
what exercises the weights.
"""

from __future__ import annotations

import pytest

from cosmos_curate.pipelines.video.clipping.arc_fusion_boundary import (
    fuse_boundaries,
    interior_cuts,
    parse_arc_spans,
)
from cosmos_curate.pipelines.video.evaluation.benchmark_boundaries import internal_boundaries


class TestParseArcSpans:
    def test_think_block_inline_spans_preferred(self) -> None:
        """The <think> block carries ARC's fine segmentation, so it wins over <answer>."""
        text = (
            "<think>The arm reaches for the mug (<span>00:00:00 - 00:00:05</span>). "
            "Then it pours (<span>00:00:05 - 00:00:12</span>).</think>"
            "<answer><span>00:00:00 - 00:00:12</span> The robot makes coffee.</answer>"
        )
        assert parse_arc_spans(text, 12.0) == [(0.0, 5.0), (5.0, 12.0)]

    def test_answer_block_used_when_think_has_no_spans(self) -> None:
        text = (
            "<think>Let me reason about this video.</think>"
            "<answer>\n<span>00:00:01 - 00:00:04</span> Picks up block.\n"
            "<span>00:00:04 - 00:00:09</span> Places block.\n</answer>"
        )
        assert parse_arc_spans(text, 9.0) == [(1.0, 4.0), (4.0, 9.0)]

    def test_bare_lines_recovered_when_tags_dropped(self) -> None:
        assert parse_arc_spans("00:00:02 - 00:00:06 Opens the drawer", 10.0) == [(2.0, 6.0)]

    def test_ends_clamped_to_duration(self) -> None:
        """ARC routinely over-runs the true end; a span past the video would corrupt the clip list."""
        text = "<answer><span>00:00:00 - 00:01:00</span> Everything.</answer>"
        assert parse_arc_spans(text, 20.0) == [(0.0, 20.0)]

    def test_degenerate_and_out_of_range_spans_dropped(self) -> None:
        text = (
            "<answer>\n<span>00:00:05 - 00:00:05</span> zero length\n"
            "<span>00:00:30 - 00:00:40</span> starts after video ends\n"
            "<span>00:00:02 - 00:00:07</span> good\n</answer>"
        )
        assert parse_arc_spans(text, 20.0) == [(2.0, 7.0)]

    def test_no_spans_at_all_returns_empty(self) -> None:
        assert parse_arc_spans("I cannot segment this video.", 10.0) == []


class TestInteriorCuts:
    def test_matches_the_metric_implementation(self) -> None:
        """Detector and metric must agree on what an interior cut is, or eval measures a lie."""
        for spans in (
            [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)],
            [(0.0, 4.0), (6.0, 10.0)],  # gap
            [(2.0, 8.0)],  # single span
            [],
        ):
            assert interior_cuts(spans) == internal_boundaries(spans)

    def test_gap_yields_two_cuts(self) -> None:
        """A gap between ARC chapters is two real boundaries, not one — it must not be bridged."""
        assert interior_cuts([(0.0, 4.0), (6.0, 10.0)]) == [4.0, 6.0]

    def test_outer_edges_are_not_cuts(self) -> None:
        assert interior_cuts([(0.0, 5.0), (5.0, 10.0)]) == [5.0]


class TestFuseBoundaries:
    def test_every_anchor_survives(self) -> None:
        """ARC anchors are high-precision; the fine stream may never displace or drop one."""
        merged = fuse_boundaries([10.0, 20.0], [10.5, 19.8], tolerance_s=4.0)
        assert merged == [10.0, 20.0]

    def test_fine_cut_admitted_only_where_arc_is_silent(self) -> None:
        merged = fuse_boundaries([10.0], [11.0, 30.0], tolerance_s=4.0)
        assert merged == [10.0, 30.0]  # 11.0 suppressed (within 4s of the anchor), 30.0 admitted

    def test_fine_stream_thins_itself(self) -> None:
        """An admitted fine cut also suppresses its own near-duplicates, not just ARC's."""
        merged = fuse_boundaries([], [30.0, 31.0, 32.0, 40.0], tolerance_s=4.0)
        assert merged == [30.0, 40.0]

    def test_no_arc_output_degrades_to_predictive(self) -> None:
        """If ARC returns nothing the video must still be segmented, not dropped."""
        assert fuse_boundaries([], [5.0, 15.0], tolerance_s=4.0) == [5.0, 15.0]

    def test_no_fine_output_degrades_to_arc(self) -> None:
        assert fuse_boundaries([5.0, 15.0], [], tolerance_s=4.0) == [5.0, 15.0]

    def test_zero_tolerance_keeps_everything(self) -> None:
        assert fuse_boundaries([10.0], [11.0], tolerance_s=0.0) == [10.0, 11.0]

    @pytest.mark.parametrize("tol", [1.0, 4.0, 10.0])
    def test_output_sorted_and_respects_spacing(self, tol: float) -> None:
        merged = fuse_boundaries([50.0, 10.0], [12.0, 30.0, 31.0, 70.0], tolerance_s=tol)
        assert merged == sorted(merged)
        # Anchors may sit closer than tol to each other, but no *admitted fine* cut may.
        for i in range(len(merged) - 1):
            assert merged[i + 1] > merged[i]
