"""Save / load darkroom recipes as JSON.

A recipe captures the control surface for a calibration pass — film, chemistry,
print filtration — so the same settings can be reloaded across frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RECIPE_VERSION = 1


def build_recipe(
    *,
    film_id: str,
    developer_id: str,
    development_minutes: float,
    contrast: float,
    grain: float,
    paper_id: str,
    print_grade: float,
    print_exposure: float,
    print_contrast: float = 0.0,
    name: str = "",
    notes: str = "",
    chemistry_mode: str = "bw",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(chemistry_mode or "bw").lower()
    if mode not in {"bw", "color"}:
        mode = "bw"
    recipe: dict[str, Any] = {
        "digital_negative_recipe_version": RECIPE_VERSION,
        "name": name or "untitled",
        "notes": notes,
        "chemistry_mode": mode,
        "film_id": str(film_id),
        "developer_id": str(developer_id),
        "development_minutes": float(development_minutes),
        "contrast": float(contrast),
        "grain": float(grain),
        "paper_id": str(paper_id),
        "print_grade": float(print_grade),
        "print_exposure": float(print_exposure),
        "print_contrast": float(print_contrast),
    }
    if extras:
        recipe["extensions"] = dict(extras)
    return recipe


def recipe_to_json(recipe: dict[str, Any]) -> str:
    return json.dumps(recipe, indent=2) + "\n"


def save_recipe(path: str | Path, recipe: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(recipe_to_json(recipe), encoding="utf-8")
    return path


def load_recipe(path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path, dict):
        data = path
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a JSON object")
    required = (
        "film_id",
        "developer_id",
        "development_minutes",
        "paper_id",
        "print_grade",
        "print_exposure",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Recipe missing: {', '.join(missing)}")
    return data
