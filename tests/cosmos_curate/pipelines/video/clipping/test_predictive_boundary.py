# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the predictive-surprise event boundary detector's algorithm core.

These exercise the detector's *behaviour* on synthetic latent trajectories with known boundaries
(abrupt change, gradual change, no change, noise) rather than merely asserting the code runs.
The GPU encoders are not involved: the core is deliberately pure numpy so it can be tested here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cosmos_curate.pipelines.video.clipping.predictive_boundary import (
    PredictiveBoundaryConfig,
    detect_from_latents,
    pick_boundaries,
    robust_z,
    surprise_signal,
)

_DIM = 32
_DT = 0.5  # seconds per latent, matching sample_fps=4 with tubelet_size=2


def _cfg(**over: object) -> PredictiveBoundaryConfig:
    base = {"history": 4, "horizons": (1, 2), "z_threshold": 2.0, "min_segment_s": 1.0}
    base.update(over)
    return PredictiveBoundaryConfig(**base)  # type: ignore[arg-type]


def _timestamps(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float32) * _DT).astype(np.float32)


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


_SCALE = 10.0  # realistic feature norm; encoders never emit vectors near the origin


def _drifting_segment(
    n: int,
    center: np.ndarray,
    *,
    rng: np.random.Generator,
    noise: float = 0.0,
    drift: float = 0.15,
) -> np.ndarray:
    """One coherent event: a latent parked near `center`, drifting slowly along a tangent.

    This mirrors what real encoders emit — features with a large, stable norm that move slowly
    within an event. (Latents that pass through the origin or grow in norm across events are not
    a real regime, and would make the cosine geometry the detector works in ill-conditioned.)
    """
    tangent = _random_unit(rng)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    seq = center[None, :] + drift * tangent[None, :] * t
    if noise:
        seq = seq + rng.normal(0.0, noise, size=seq.shape).astype(np.float32)
    return (seq * _SCALE).astype(np.float32)


# ── robust_z ──────────────────────────────────────────────────────────────────────────────────


def test_robust_z_is_not_masked_by_the_outlier_it_must_find() -> None:
    """The key property motivating median/MAD over mean/std."""
    x = np.concatenate([np.full(40, 0.1, dtype=np.float32), np.array([5.0], dtype=np.float32)])
    x[:40] += np.linspace(-0.01, 0.01, 40, dtype=np.float32)  # tiny spread so MAD > 0

    z_robust = robust_z(x)
    z_naive = (x - x.mean()) / x.std()

    assert z_robust[-1] > 50.0, "median/MAD must expose the outlier"
    assert z_naive[-1] < 7.0, "mean/std is inflated by the outlier itself"
    assert z_robust[-1] > z_naive[-1]


def test_robust_z_constant_signal_yields_zeros_not_nan() -> None:
    """A frozen signal has no events; it must score zero, not NaN."""
    assert np.allclose(robust_z(np.full(20, 3.0, dtype=np.float32)), 0.0)


def test_robust_z_ignores_nans() -> None:
    """The anticipator's warm-up NaNs must not poison the statistics."""
    x = np.array([np.nan, 1.0, 1.1, 0.9, 1.0, 8.0], dtype=np.float32)
    z = robust_z(x)
    assert np.isfinite(z).all()
    assert z[-1] > z[1]


# ── surprise_signal ───────────────────────────────────────────────────────────────────────────


def test_surprise_is_undefined_in_the_warmup_window() -> None:
    """Before `history + max(horizons) - 1` latents there is nothing to extrapolate from."""
    rng = np.random.default_rng(0)
    lat = _drifting_segment(30, _random_unit(rng), rng=rng)
    cfg = _cfg()
    sig = surprise_signal(lat, horizons=cfg.horizons, history=cfg.history)
    start = cfg.history + max(cfg.horizons) - 1
    assert np.isnan(sig[:start]).all(), "no prediction is possible before enough history"
    assert np.isfinite(sig[start:]).all()


def test_surprise_signal_too_short_is_all_nan() -> None:
    """Too few latents to fit any history: nothing is predictable."""
    sig = surprise_signal(np.zeros((3, _DIM), dtype=np.float32), horizons=(1, 2), history=4)
    assert np.isnan(sig).all()


def test_surprise_peaks_at_an_abrupt_change() -> None:
    """A hard transition (the latent jumps to an unrelated direction) must spike the signal."""
    rng = np.random.default_rng(1)
    a, b = _random_unit(rng), _random_unit(rng)
    lat = np.concatenate([_drifting_segment(20, a, rng=rng), _drifting_segment(20, b, rng=rng)])
    cfg = _cfg()
    sig = surprise_signal(lat, horizons=cfg.horizons, history=cfg.history)
    peak = int(np.nanargmax(sig))
    assert abs(peak - 20) <= 2, f"peak at {peak}, expected near the change at 20"


def test_surprise_stays_flat_when_dynamics_are_constant() -> None:
    """A single coherent event: the anticipator tracks it, so nothing should look surprising."""
    rng = np.random.default_rng(2)
    lat = _drifting_segment(40, _random_unit(rng), rng=rng)
    cfg = _cfg()
    sig = surprise_signal(lat, horizons=cfg.horizons, history=cfg.history)
    finite = sig[np.isfinite(sig)]
    assert np.nanmax(np.abs(finite)) < cfg.z_threshold, "no boundary should be implied"


# ── pick_boundaries ───────────────────────────────────────────────────────────────────────────


def test_pick_boundaries_suppresses_neighbours_and_keeps_the_strongest() -> None:
    """Ridge echoes near a strong peak are suppressed; a distant peak survives."""
    fused = np.zeros(30, dtype=np.float32)
    fused[10] = 5.0  # strongest
    fused[12] = 4.0  # within min_gap of the peak -> must be suppressed
    fused[25] = 4.5  # far away -> kept
    got = pick_boundaries(
        fused, _timestamps(30), z_threshold=2.0, min_gap=4, duration_s=30 * _DT, min_edge_s=1.0
    )
    assert got == [round(10 * _DT, 3), round(25 * _DT, 3)]


def test_pick_boundaries_drops_peaks_at_the_video_edges() -> None:
    """A cut at t~0 or t~duration would make a degenerate clip."""
    fused = np.zeros(30, dtype=np.float32)
    fused[1] = 9.0  # start edge
    fused[28] = 9.0  # end edge
    fused[15] = 9.0  # interior
    got = pick_boundaries(
        fused, _timestamps(30), z_threshold=2.0, min_gap=2, duration_s=30 * _DT, min_edge_s=1.0
    )
    assert got == [round(15 * _DT, 3)]


def test_pick_boundaries_respects_threshold() -> None:
    """A sub-threshold peak is not a boundary."""
    fused = np.zeros(20, dtype=np.float32)
    fused[10] = 1.5
    assert pick_boundaries(fused, _timestamps(20), z_threshold=2.0, min_gap=2, duration_s=10.0, min_edge_s=1.0) == []


def test_pick_boundaries_on_degenerate_input() -> None:
    """Too-short and all-NaN signals return no boundaries rather than raising."""
    too_short = np.zeros(2, dtype=np.float32)
    assert pick_boundaries(
        too_short, _timestamps(2), z_threshold=2.0, min_gap=1, duration_s=1.0, min_edge_s=0.1
    ) == []
    all_nan = np.full(10, np.nan, dtype=np.float32)
    assert pick_boundaries(all_nan, _timestamps(10), z_threshold=2.0, min_gap=1, duration_s=5.0, min_edge_s=0.1) == []


# ── End-to-end over latents ───────────────────────────────────────────────────────────────────


def _three_event_sequence(rng: np.random.Generator, *, noise: float = 0.0, seg: int = 24) -> np.ndarray:
    """Three coherent events; true boundaries at latent indices `seg` and `2*seg`."""
    segs = [_drifting_segment(seg, _random_unit(rng), rng=rng, noise=noise) for _ in range(3)]
    return np.concatenate(segs).astype(np.float32)


def test_detects_both_boundaries_of_a_three_event_video() -> None:
    """The headline behaviour: exactly the two true event boundaries, well localised."""
    rng = np.random.default_rng(3)
    seg = 24
    lat = _three_event_sequence(rng, seg=seg)
    ts = _timestamps(lat.shape[0])
    got = detect_from_latents([(lat, ts)], _cfg(), duration_s=float(ts[-1]))

    expected = [seg * _DT, 2 * seg * _DT]
    assert len(got) == 2, f"expected 2 boundaries, got {got}"
    for g, e in zip(sorted(got), expected, strict=True):
        assert abs(g - e) <= 1.0, f"boundary {g} too far from {e}"


def test_robust_to_latent_noise() -> None:
    """Boundaries must survive per-frame feature noise, which is what real encoders produce."""
    rng = np.random.default_rng(4)
    seg = 24
    lat = _three_event_sequence(rng, noise=0.02, seg=seg)
    ts = _timestamps(lat.shape[0])
    got = detect_from_latents([(lat, ts)], _cfg(), duration_s=float(ts[-1]))
    assert len(got) == 2
    for g, e in zip(sorted(got), [seg * _DT, 2 * seg * _DT], strict=True):
        assert abs(g - e) <= 1.0


def test_gradual_transition_is_detected_by_the_long_horizon() -> None:
    """A slow cross-fade between two events: 1-step prediction tracks it, longer horizons do not.

    This is why the signal is multi-horizon; with only horizon=1 the ramp is nearly invisible.
    """
    rng = np.random.default_rng(5)
    a, b = _random_unit(rng), _random_unit(rng)
    hold, ramp = 24, 12
    seq = [np.tile(a, (hold, 1))]
    w = np.linspace(0.0, 1.0, ramp, dtype=np.float32)[:, None]
    seq.append((1 - w) * a[None, :] + w * b[None, :])
    seq.append(np.tile(b, (hold, 1)))
    noise = rng.normal(0, 0.005, (hold * 2 + ramp, _DIM)).astype(np.float32)
    lat = np.concatenate(seq).astype(np.float32) + noise

    long_h = surprise_signal(lat, horizons=(4,), history=4)
    short_h = surprise_signal(lat, horizons=(1,), history=4)
    ramp_slice = slice(hold, hold + ramp)
    assert np.nanmax(long_h[ramp_slice]) > np.nanmax(short_h[ramp_slice]), (
        "the long horizon must be more sensitive to a gradual transition than the 1-step one"
    )


def test_no_boundaries_on_a_static_video() -> None:
    """Constant latents (a frozen shot) must yield no boundaries, not a crash or spurious cuts."""
    lat = np.tile(np.ones(_DIM, dtype=np.float32), (40, 1))
    ts = _timestamps(40)
    assert detect_from_latents([(lat, ts)], _cfg(), duration_s=float(ts[-1])) == []


def test_min_segment_s_enforces_spacing() -> None:
    """Detected boundaries are never closer together than min_segment_s."""
    rng = np.random.default_rng(6)
    lat = _three_event_sequence(rng, seg=6)  # boundaries 3 s apart at dt=0.5
    ts = _timestamps(lat.shape[0])
    got = detect_from_latents([(lat, ts)], _cfg(min_segment_s=5.0), duration_s=float(ts[-1]))
    for a, b in itertools.pairwise(got):
        assert b - a >= 5.0 - 1e-6


def test_empty_and_short_inputs_return_no_boundaries() -> None:
    """Degenerate inputs degrade to [] — the stage then emits the whole video as one clip."""
    cfg = _cfg()
    assert detect_from_latents([], cfg, duration_s=10.0) == []
    tiny = np.zeros((2, _DIM), dtype=np.float32)
    assert detect_from_latents([(tiny, _timestamps(2))], cfg, duration_s=1.0) == []


def test_fusing_two_streams_agrees_with_a_single_stream_when_identical() -> None:
    """Fusion is a weighted mean: duplicating a stream must not change the result."""
    rng = np.random.default_rng(7)
    lat = _three_event_sequence(rng)
    ts = _timestamps(lat.shape[0])
    cfg_one = _cfg(stream_weights=(1.0,))
    cfg_two = _cfg(stream_weights=(1.0, 1.0))
    one = detect_from_latents([(lat, ts)], cfg_one, duration_s=float(ts[-1]))
    two = detect_from_latents([(lat, ts), (lat, ts)], cfg_two, duration_s=float(ts[-1]))
    assert one == two


def test_vjepa2_token_layout_matches_the_encoders_reshape() -> None:
    """Lock the contract `VJepa2TubeletEncoder` depends on: tokens are ordered (tubelet, patch).

    The encoder reshapes `last_hidden_state` to `(t_clip, spatial, dim)` and pools over space. If
    V-JEPA 2 ever emitted space-major tokens that reshape would silently blend time into space and
    every latent would be wrong — with no exception raised. A tiny randomly-initialised model is
    enough to assert the layout: real weights are irrelevant to token ordering.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    cfg = transformers.VJEPA2Config(
        crop_size=64, patch_size=16, tubelet_size=2, hidden_size=32,
        num_hidden_layers=1, num_attention_heads=2, mlp_ratio=1.0,
        pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2,
    )
    model = transformers.VJEPA2Model(cfg).eval()

    n_frames, tub = 8, cfg.tubelet_size
    patches = (cfg.crop_size // cfg.patch_size) ** 2
    # A step video: first half black, second half white -> one hard temporal change.
    video = torch.zeros(1, n_frames, 3, cfg.crop_size, cfg.crop_size)
    video[:, n_frames // 2 :] = 1.0
    with torch.no_grad():
        tokens = model(pixel_values_videos=video, skip_predictor=True).last_hidden_state

    t_clip = n_frames // tub
    assert tokens.shape[1] == t_clip * patches, "token count is not (frames/tubelet) * patches"

    pooled = tokens[0].reshape(t_clip, tokens.shape[1] // t_clip, tokens.shape[2]).mean(1).numpy()
    first, second = pooled[: t_clip // 2], pooled[t_clip // 2 :]
    within = max(np.abs(first - first[0]).max(), np.abs(second - second[0]).max())
    across = float(np.abs(first.mean(0) - second.mean(0)).max())
    assert within < 1e-4, "tubelets inside one half differ -> tokens are not time-major"
    assert across > 1e-2, "the two halves collapsed -> the reshape is mixing time with space"


@pytest.mark.parametrize("z_threshold", [1.0, 2.0, 3.0])
def test_higher_threshold_never_yields_more_boundaries(z_threshold: float) -> None:
    """Monotonicity: the threshold behaves as a precision/recall dial, as documented."""
    rng = np.random.default_rng(8)
    lat = _three_event_sequence(rng, noise=0.03)
    ts = _timestamps(lat.shape[0])
    loose = detect_from_latents([(lat, ts)], _cfg(z_threshold=0.5), duration_s=float(ts[-1]))
    got = detect_from_latents([(lat, ts)], _cfg(z_threshold=z_threshold), duration_s=float(ts[-1]))
    assert len(got) <= len(loose)
