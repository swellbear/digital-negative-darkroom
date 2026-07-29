"""Digital Negative data model and sidecar I/O."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_metadata(
    *,
    source: dict[str, Any] | None = None,
    film_profile: dict[str, Any] | None = None,
    process_seed: int | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    seed = process_seed if process_seed is not None else int(uuid.uuid4().int % (2**31 - 1))
    return {
        "digital_negative_version": "1.0",
        "uuid": str(uuid.uuid4()),
        "created": now,
        "modified": now,
        "source": source
        or {
            "original_filename": "",
            "camera_make": "",
            "camera_model": "",
            "iso": 0,
            "shutter_speed": "",
            "aperture": "",
            "datetime_original": "",
            "raw_hash": "",
        },
        "ingest": {
            "white_balance": {"method": "as_shot", "temperature": 0, "tint": 0},
            "orientation": 1,
        },
        "film_profile": film_profile
        or {
            "id": "",
            "name": "",
            "type": "bw",
            "version": "1.0",
            "iso": 0,
        },
        "development": {
            "enabled": True,
            "developer_id": "standard",
            "developer_name": "Standard",
            "relative_time": 1.0,
            "contrast_modifier": 0.0,
            "grain_strength": 1.0,
            "notes": "",
        },
        "print": {
            "enabled": False,
            "paper_id": "mg-standard",
            "paper_name": "Multigrade Standard",
            "filtration": {"type": "multigrade", "grade": 2.5, "values": {}},
            "overall_exposure": 0.0,
            "contrast": 0.0,
            "dodge_burn": [],
        },
        "process_seed": seed,
        "history": [],
        "ui_state": {"current_stage": "development", "committed_stages": ["ingest"]},
        "extensions": {},
    }


@dataclass
class DigitalNegative:
    """Scene-referred linear image + process metadata.

    Image payload is float32 luminance (H, W) or RGB (H, W, 3),
    near-linear / scene-referred. Creative decisions live in metadata.
    """

    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=default_metadata)

    def __post_init__(self) -> None:
        if self.image.dtype != np.float32:
            self.image = self.image.astype(np.float32)
        if self.image.ndim not in (2, 3):
            raise ValueError("Digital Negative image must be HxW or HxWx3")

    @property
    def uuid(self) -> str:
        return str(self.metadata["uuid"])

    def to_luminance(self) -> np.ndarray:
        if self.image.ndim == 2:
            return self.image
        # Rec. 709 luma coefficients on linear light
        rgb = self.image
        return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
            np.float32
        )

    def touch(self) -> None:
        self.metadata["modified"] = _utc_now()

    def save(self, directory: str | Path, stem: str | None = None) -> tuple[Path, Path]:
        """Save image as 16-bit TIFF + JSON sidecar."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = stem or self.uuid
        tiff_path = directory / f"{stem}.tiff"
        json_path = directory / f"{stem}.json"

        arr = np.clip(self.image, 0.0, None)
        # Encode with a soft headroom so bright highlights survive 16-bit storage
        peak = float(np.percentile(arr, 99.9)) if arr.size else 1.0
        scale = peak if peak > 1e-8 else 1.0
        norm = np.clip(arr / scale, 0.0, 1.0)
        u16 = (norm * 65535.0 + 0.5).astype(np.uint16)
        iio.imwrite(tiff_path, u16)

        self.metadata.setdefault("extensions", {})["storage_peak"] = scale
        self.touch()
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            f.write("\n")
        return tiff_path, json_path

    @classmethod
    def load(cls, tiff_path: str | Path, json_path: str | Path | None = None) -> "DigitalNegative":
        tiff_path = Path(tiff_path)
        json_path = Path(json_path) if json_path else tiff_path.with_suffix(".json")
        u16 = np.asarray(iio.imread(tiff_path))
        if u16.dtype != np.uint16:
            u16 = u16.astype(np.uint16)
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        peak = float(meta.get("extensions", {}).get("storage_peak", 1.0))
        image = (u16.astype(np.float32) / 65535.0) * peak
        return cls(image=image, metadata=meta)
