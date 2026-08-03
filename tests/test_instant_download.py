"""Instant Commit pull must expose a finished-card download."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_instant_card_package_writer():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    path = ROOT / "scripts" / "run_darkroom_ui.py"
    spec = importlib.util.spec_from_file_location("run_darkroom_ui", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    class _Prof:
        id = "polaroid-600-instant-v1"
        name = "Polaroid 600"

    class _DN:
        metadata = {"source": {"original_filename": "frame.jpg"}}

    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:] = (210, 180, 140)
    zpath = mod._write_instant_card_package(rgb, _DN(), _Prof())
    assert Path(zpath).is_file()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.startswith("card/") and n.endswith(".png") for n in names)


def test_instant_commit_shows_download_card():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "_write_instant_card_package" in source
    assert 'value="⇣ Download card"' in source
    assert 'elem_id="download_row"' in source
    # Download strip must not live inside the Print drawer (hidden in Instant).
    print_chunk = source.split('elem_id="drawer_print"')[1].split('elem_id="drawer_frame"')[0]
    assert "download_trigger" not in print_chunk
    assert 'mode === \'card\' ? \'negative\'' in source or 'mode === "card" ? "negative"' in source


def test_download_labels_include_card():
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    assert "card: 'Finished card'" in source or 'card: "Finished card"' in source


def test_instant_package_uses_full_res_card_not_live_max():
    """Commit pull packages positive_preview; LIVE_MAX_SIDE is preview-only."""
    source = (ROOT / "scripts" / "run_darkroom_ui.py").read_text(encoding="utf-8")
    # Full card for ZIP
    assert "card_full = _to_rgb_u8(development.positive_preview)" in source
    assert "_write_instant_card_package(card_pkg" in source
    # Live preview still downscaled
    assert (
        "live_view = _downscale_rgb(\n"
        "            _to_rgb_u8(development.positive_preview), LIVE_MAX_SIDE"
    ) in source or (
        "_downscale_rgb(\n"
        "            _to_rgb_u8(development.positive_preview), LIVE_MAX_SIDE"
    ) in source
    # Must not package the downscaled live view
    assert "_write_instant_card_package(live_view" not in source
    assert "_write_instant_card_package(live_rgb" not in source
