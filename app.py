#!/usr/bin/env python3
"""Hugging Face Spaces / Gradio entrypoint for Digital Negative Darkroom."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_darkroom_ui import build_ui  # noqa: E402

demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch()
