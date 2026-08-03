"""Live hq develop/print must not bake grain on a LIVE_MAX_SIDE stride proxy."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hq_live_develop_uses_full_or_inspect_cap_not_live_proxy():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    chunk = source.split("def _run_live_develop_then_print(")[1].split("def live_preview(")[0]
    assert "INSPECT_MAX_SIDE" in chunk
    assert "proxy_drag" in chunk
    # Hq working set is the full DN (or inspect-capped), not the LIVE proxy.
    assert "working = src" in chunk
    assert "_downscale_rgb_hq(live_rgb, max_side)" in chunk
    assert "bake grain on the full" in chunk


def test_locked_print_hq_prints_full_transmittance():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    chunk = source.split("if _locked(state, \"development\"):")[1].split(
        "# Develop unlocked:"
    )[0]
    assert "_print_transmittance(dev_full)" in chunk
    assert "_downscale_rgb_hq(_to_rgb_u8(result.preview), max_side)" in chunk
