#!/usr/bin/env python3
"""CLI: ingest → Digital Negative → develop → print → previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.pipeline import run_darkroom_pipeline  # noqa: E402


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
    parser.add_argument("--film", default="hp5-plus-v1", help="Film profile id")
    parser.add_argument("--paper", default="mg-standard", help="Paper profile id")
    parser.add_argument(
        "--developer",
        default="standard",
        choices=["standard", "high_definition", "high_energy"],
    )
    parser.add_argument("--relative-time", type=float, default=1.0, help="Push/pull factor")
    parser.add_argument("--contrast", type=float, default=0.0, help="Development contrast")
    parser.add_argument("--grain", type=float, default=1.0, help="Grain strength")
    parser.add_argument("--print-exposure", type=float, default=0.0, help="Print exposure stops")
    parser.add_argument("--print-grade", type=float, default=2.5, help="Multigrade grade 0-5")
    parser.add_argument("--print-contrast", type=float, default=0.0, help="Print contrast nudge")
    parser.add_argument("--no-print", action="store_true", help="Skip print stage")
    args = parser.parse_args()

    artifacts = run_darkroom_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        film_id=args.film,
        paper_id=args.paper,
        developer_id=args.developer,
        relative_time=args.relative_time,
        contrast_modifier=args.contrast,
        grain_strength=args.grain,
        print_exposure=args.print_exposure,
        print_grade=args.print_grade,
        print_contrast=args.print_contrast,
        do_print=not args.no_print,
    )

    print("Darkroom pipeline complete.")
    print(json.dumps(artifacts.stats, indent=2))
    print(f"Digital Negative TIFF: {artifacts.dn_tiff}")
    print(f"Metadata sidecar:      {artifacts.dn_json}")
    print(f"Developed preview:     {artifacts.developed_preview}")
    if artifacts.print_preview:
        print(f"Print preview:         {artifacts.print_preview}")
    print(f"Comparison strip:      {artifacts.comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
