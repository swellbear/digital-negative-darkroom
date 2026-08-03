"""Spot meter must re-sample when the live print changes (not only on hover)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pack_preview_includes_spot_readout():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "spot_md = read_spot((state or {}).get(\"spot_pos\"" in source
    assert "spot_readout," in source.split("preview_outputs = [")[1].split("]")[0]
    assert "fn=remember_spot" in source


def test_remember_spot_persists_pointer_and_read_spot_uses_print_draft():
    import importlib.util
    import numpy as np

    from digital_negative.analysis import spot_at, spot_markdown
    from digital_negative.print_engine import PrintResult

    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui_spot", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    h, w = 32, 24
    # Two different print maps → same pointer must read different Zone/D.
    refl_a = np.full((h, w), 0.25, dtype=np.float32)
    dens_a = (-np.log10(np.maximum(refl_a, 1e-6))).astype(np.float32)
    refl_b = np.full((h, w), 0.55, dtype=np.float32)
    dens_b = (-np.log10(np.maximum(refl_b, 1e-6))).astype(np.float32)

    def _draft(refl, dens):
        preview = np.clip(refl, 0, 1)
        return PrintResult(
            preview=preview,
            print_density=dens,
            reflectance=refl,
        )

    class _DN:
        metadata = {"ui_state": {"locked_stages": []}}

    state = {
        "dn": _DN(),
        "spot_pos": "0.5000,0.5000",
        "print_draft": _draft(refl_a, dens_a),
        "live_rgb": np.zeros((h, w, 3), dtype=np.float32),
        "viewer_mode": "live",
    }
    md_a = mod.read_spot(state["spot_pos"], state)
    expect_a = spot_markdown(spot_at(refl_a, dens_a, 0.5, 0.5))
    assert md_a == expect_a

    state["print_draft"] = _draft(refl_b, dens_b)
    packed = mod._pack_preview(
        state["live_rgb"], None, None, None, "status", state, mark_dirty=False
    )
    assert packed[-2] == spot_markdown(spot_at(refl_b, dens_b, 0.5, 0.5))
    assert packed[-2] != md_a

    md, state2 = mod.remember_spot("0.2500,0.7500", state)
    assert state2["spot_pos"] == "0.2500,0.7500"
    assert md == spot_markdown(spot_at(refl_b, dens_b, 0.25, 0.75))
