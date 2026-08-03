"""Commit Develop must not change the on-screen live print."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_ui():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _state_from(outputs):
    return next(x for x in outputs if isinstance(x, dict) and "roll" in x)


def _live_state(mod, state, controls):
    packed = mod.live_preview_high(
        controls["chemistry_mode"],
        controls["film_id"],
        controls["developer_id"],
        controls["development_minutes"],
        controls["contrast"],
        controls["grain"],
        controls["exposure_index"],
        controls["contrast_filter"],
        controls["scene_exposure"],
        controls["halation"],
        controls["paper_id"],
        controls["print_exposure"],
        controls["print_grade"],
        controls["print_contrast"],
        controls["split_grade"],
        controls["soft_grade"],
        controls["hard_grade"],
        controls["soft_seconds"],
        controls["hard_seconds"],
        controls["test_strips"],
        controls["test_bands"],
        controls["test_stops"],
        controls["flash_stops"],
        controls["dry_down"],
        controls["tone"],
        controls["border_frac"],
        controls["cc_cyan"],
        controls["cc_magenta"],
        controls["cc_yellow"],
        controls.get("process_temp_c", 21.0),
        controls.get("instant_chroma", 1.0),
        controls.get("instant_warmth", 0.0),
        controls.get("instant_border", True),
        state,
    )
    return next(x for x in packed if isinstance(x, dict) and "live_rgb" in x)


def test_commit_develop_keeps_live_print_shape_and_pixels(tmp_path):
    """Large frames used to double-stride after lock (1344×1008 → 1008×756)."""
    mod = _load_ui()
    # Tall 4K-ish frame: proxy lands on LIVE_MAX_SIDE stride (same as user report).
    h, w = 4032, 3024
    path = tmp_path / "tall_4k.png"
    rng = np.random.default_rng(1)
    Image.fromarray((rng.random((h, w, 3)) * 180 + 30).astype(np.uint8)).save(path)

    state = _state_from(mod.commit_ingest(None, str(path), None))
    controls = mod._default_controls_dict()
    controls["print_exposure"] = 15.5

    before_state = _live_state(mod, state, controls)
    before = np.asarray(before_state["live_rgb"])
    assert max(before.shape[:2]) <= mod.LIVE_MAX_SIDE

    committed = next(
        x
        for x in mod.commit_develop(
            controls["film_id"],
            controls["developer_id"],
            controls["development_minutes"],
            controls["contrast"],
            controls["grain"],
            controls["exposure_index"],
            controls["contrast_filter"],
            controls["scene_exposure"],
            controls["halation"],
            False,
            "4",
            before_state,
        )
        if isinstance(x, dict) and x.get("development") is not None
    )
    after_state = _live_state(mod, committed, controls)
    after = np.asarray(after_state["live_rgb"])

    assert after.shape == before.shape
    assert np.array_equal(before, after)
