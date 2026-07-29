"""Paper profile loading for the print stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperProfile:
    id: str
    name: str
    version: str
    type: str
    dmax: float
    dmin: float
    grades: dict[str, dict[str, float]]
    default_grade: float
    raw: dict[str, Any]

    def grade_params(self, grade: float) -> dict[str, float]:
        """Interpolate grade parameters between nearest defined grades."""
        keys = sorted(float(k) for k in self.grades)
        if grade <= keys[0]:
            return dict(self.grades[self._key(keys[0])])
        if grade >= keys[-1]:
            return dict(self.grades[self._key(keys[-1])])

        for lo, hi in zip(keys, keys[1:]):
            if lo <= grade <= hi:
                t = (grade - lo) / (hi - lo) if hi != lo else 0.0
                a = self.grades[self._key(lo)]
                b = self.grades[self._key(hi)]
                return {k: (1.0 - t) * float(a[k]) + t * float(b[k]) for k in a}
        return dict(self.grades[self._key(self.default_grade)])

    @staticmethod
    def _key(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value)


def load_paper_profile(path: str | Path) -> PaperProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return PaperProfile(
        id=data["id"],
        name=data["name"],
        version=data["version"],
        type=data["type"],
        dmax=float(data["dmax"]),
        dmin=float(data["dmin"]),
        grades={str(k): dict(v) for k, v in data["grades"].items()},
        default_grade=float(data.get("default_grade", 2.5)),
        raw=data,
    )


def default_papers_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "profiles" / "papers"
