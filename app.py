#!/usr/bin/env python3
"""Hugging Face Spaces / Gradio entrypoint for Digital Negative Darkroom."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _ensure_sample_raws() -> None:
    """Fetch public sample raws when the Space image has none checked in."""
    dest = ROOT / "samples" / "raws"
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.*")):
        return
    script = ROOT / "samples" / "fetch_raws.sh"
    if not script.is_file():
        return
    try:
        subprocess.run(
            ["bash", str(script)],
            cwd=str(ROOT),
            check=False,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001 — Space boot must continue on synthetic-only
        print(f"[app] sample fetch skipped: {exc}", file=sys.stderr)


_ensure_sample_raws()

from run_darkroom_ui import build_ui  # noqa: E402

demo = build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
    )
