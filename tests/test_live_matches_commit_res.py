"""Live hq and Commit must share the film-true full-negative path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hq_live_develop_uses_live_proxy_for_snappy_film_swaps():
    """Interactive hq must not bake the full/inspect frame (hangs on film change)."""
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    chunk = source.split("def _run_live_develop_then_print(")[1].split("def live_preview(")[0]
    assert "proxy_drag" in chunk
    assert 'or _proxy_dn(state["dn"], LIVE_MAX_SIDE)' in chunk
    assert "working = src" not in chunk
    assert "_downscale_rgb(_to_rgb_u8(printed.preview), max_side)" in chunk
    assert "fn=on_film_change_and_preview" in source
    assert "fn=on_developer_change_and_preview" in source
    assert "_CHEM_UI_BATCH" in source
    assert "cancels=[film_preview_evt]" not in source


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
