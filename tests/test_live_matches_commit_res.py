"""Live hq and Commit must share the film-true full-negative path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hq_live_develop_uses_full_or_inspect_cap_not_live_proxy():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    chunk = source.split("def _run_live_develop_then_print(")[1].split("def live_preview(")[0]
    assert "INSPECT_MAX_SIDE" in chunk
    assert "proxy_drag" in chunk
    assert "working = src" in chunk
    # Viewer fit must preserve grain (stride), not Lanczos-clean it.
    assert "_downscale_rgb(_to_rgb_u8(printed.preview), max_side)" in chunk
    assert "_downscale_rgb_hq" not in chunk


def test_locked_print_and_commit_use_stride_fit_not_lanczos():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "def _downscale_rgb_hq" not in source
    commit = source.split("def commit_print(")[1].split("def _unlock_stage(")[0]
    assert "live_rgb = _downscale_rgb(print_full, LIVE_MAX_SIDE)" in commit
    locked = source.split('if _locked(state, "development"):')[1].split(
        "# Develop unlocked:"
    )[0]
    assert "_print_transmittance(dev_full)" in locked
    assert "_downscale_rgb(_to_rgb_u8(result.preview), max_side)" in locked
