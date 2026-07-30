"""Camera roll: multi-ingest, switch frames, remove."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scene_linear_srgb.png"


def _load_ui():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_from(outputs):
    return next(x for x in outputs if isinstance(x, dict) and "roll" in x)


def test_collect_input_paths_multi_and_sample():
    mod = _load_ui()
    assert mod._collect_input_paths(["/a.jpg", "/b.jpg"], "/sample.nef") == [
        "/a.jpg",
        "/b.jpg",
    ]
    assert mod._collect_input_paths(None, "/sample.nef") == ["/sample.nef"]
    assert mod._collect_input_paths(None, None) == []
    assert mod._resolve_input(["/a.jpg", "/b.jpg"], "/sample.nef") == "/a.jpg"


def test_camera_roll_add_switch_remove():
    mod = _load_ui()
    path = str(FIXTURE)
    outs = mod.commit_ingest(None, [path, path], None)
    state = _state_from(outs)
    assert len(state["roll"]) == 2
    assert state["roll_index"] == 1
    assert state["dn"] is not None

    outs = mod.commit_ingest(path, None, state)
    state = _state_from(outs)
    assert len(state["roll"]) == 3
    assert state["roll_index"] == 2

    outs = mod.select_roll_frame(state, type("E", (), {"index": 0})())
    state = _state_from(outs)
    assert state["roll_index"] == 0
    assert state["dn"] is state["roll"][0]["dn"]

    outs = mod.remove_from_roll(state)
    state = _state_from(outs)
    assert len(state["roll"]) == 2
    assert state["roll_index"] == 0

    while state.get("dn") is not None:
        outs = mod.remove_from_roll(state)
        state = _state_from(outs)
    assert state["roll"] == []
    assert state["roll_index"] == -1


def test_develop_preserves_camera_roll():
    mod = _load_ui()
    path = str(FIXTURE)
    state = _state_from(mod.commit_ingest(None, [path, path], None))
    assert len(state["roll"]) == 2

    film_id = mod.FILM_CHOICES[0][1]
    chem_id = mod._INIT_DEV_ID
    outs = mod.commit_develop(
        film_id,
        chem_id,
        mod._INIT_TNORM,
        0.0,
        1.0,
        400,
        "none",
        0.01,
        0.0,
        state,
    )
    # state is near the end of commit_develop outputs
    new_state = next(x for x in outs if isinstance(x, dict) and x.get("dn") is not None)
    assert "roll" in new_state
    assert len(new_state["roll"]) == 2
    assert new_state["roll_index"] == 1
    assert "development" in mod._locks(new_state)
