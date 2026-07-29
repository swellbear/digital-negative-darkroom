#!/usr/bin/env python3
"""Technical spike: raw/image → Digital Negative → characteristic curve → preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.pipeline import run_spike_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Camera raw or image path. Omit to use the built-in synthetic scene.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(ROOT / "output"),
        help="Directory for TIFF/JSON/previews (default: ./output)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Path to film profile JSON (default: profiles/films/hp5-plus-v1.json)",
    )
    parser.add_argument("--relative-time", type=float, default=1.0, help="Push/pull factor")
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.0,
        help="Contrast modifier for the straight-line slope",
    )
    args = parser.parse_args()

    artifacts = run_spike_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        profile_path=args.profile,
        relative_time=args.relative_time,
        contrast_modifier=args.contrast,
    )

    print("Spike complete.")
    print(json.dumps(artifacts.stats, indent=2))
    print(f"Digital Negative TIFF: {artifacts.dn_tiff}")
    print(f"Metadata sidecar:      {artifacts.dn_json}")
    print(f"Density preview:       {artifacts.density_preview}")
    print(f"Positive preview:      {artifacts.positive_preview}")
    print(f"Comparison strip:      {artifacts.comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
