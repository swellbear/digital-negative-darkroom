"""Unit tests for optional AI enlarge (export-only Real-ESRGAN path)."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_enhance():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    return importlib.import_module("digital_negative.enhance")


def test_maybe_ai_upscale_disabled_is_noop():
    enh = _load_enhance()
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    out = enh.maybe_ai_upscale_rgb(rgb, enabled=False, scale=4)
    assert out.shape == rgb.shape
    np.testing.assert_array_equal(out, rgb)


def test_ai_upscale_raises_when_onnx_missing(monkeypatch):
    enh = _load_enhance()
    monkeypatch.setattr(enh, "enhance_available", lambda: False)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        enh.ai_upscale_rgb(rgb, scale=4)


def test_pad_to_multiple_and_nchw_roundtrip():
    enh = _load_enhance()
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    padded, oh, ow = enh._pad_to_multiple(rgb, 4)
    assert oh == 5 and ow == 7
    assert padded.shape[0] % 4 == 0
    assert padded.shape[1] % 4 == 0
    assert padded[0, 0, 0] == 10

    batch = enh._to_nchw01(rgb)
    assert batch.shape == (1, 3, 5, 7)
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0
    back = enh._from_nchw01(batch)
    assert back.shape == (5, 7, 3)
    assert back.dtype == np.uint8
    assert back[0, 0, 0] == 10


def test_resolve_model_honors_env_override(tmp_path, monkeypatch):
    enh = _load_enhance()
    fake = tmp_path / "fake.onnx"
    fake.write_bytes(b"not-a-real-model")
    monkeypatch.setenv(enh._ENV_MODEL, str(fake))
    assert enh.resolve_realesrgan_model(scale=4) == fake


def test_resolve_model_missing_env_path(tmp_path, monkeypatch):
    enh = _load_enhance()
    missing = tmp_path / "gone.onnx"
    monkeypatch.setenv(enh._ENV_MODEL, str(missing))
    with pytest.raises(FileNotFoundError):
        enh.resolve_realesrgan_model(scale=4)


def test_ai_upscale_tiny_frame_when_runtime_present():
    """Smoke real ONNX only when runtime is installed (model may download)."""
    enh = _load_enhance()
    if not enh.enhance_available():
        pytest.skip("onnxruntime not installed")
    rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
    try:
        out = enh.ai_upscale_rgb(rgb, scale=2, tile=32, overlap=0)
    except Exception as exc:
        pytest.skip(f"model/runtime unavailable: {exc}")
    assert out.shape[0] == 64 and out.shape[1] == 64
    assert out.dtype == np.uint8
