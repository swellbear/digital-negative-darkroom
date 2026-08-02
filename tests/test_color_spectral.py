"""Spectral color chemistry: C-41 / E-6 / RA-4 + B&W isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digital_negative.color_development import color_negative_lightbox_preview
from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import print_negative
from digital_negative.spectral import (
    N_WAVELENGTHS,
    chemistry_mode_for_film_type,
    spectral_to_xyz,
    xyz_to_spectral,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wavelength_grid_and_xyz_roundtrip():
    xyz = np.full((16, 16, 3), 0.18, dtype=np.float32)
    spec = xyz_to_spectral(xyz)
    assert spec.shape == (16, 16, N_WAVELENGTHS)
    assert float(spec.min()) >= 0.0
    back = spectral_to_xyz(spec)
    assert abs(float(back[..., 1].mean()) - 0.18) < 0.05


def test_profile_lists_split_by_chemistry_mode():
    bw_films = list_film_profiles(chemistry_mode="bw")
    color_films = list_film_profiles(chemistry_mode="color")
    assert bw_films
    assert color_films
    bw_ids = {p.stem for p in bw_films}
    color_ids = {p.stem for p in color_films}
    assert "hp5-plus-v1" in bw_ids
    assert "tmax-400-v1" in bw_ids
    assert "delta-400-v1" in bw_ids
    assert "acros-100-ii-v1" in bw_ids
    assert "portra-400-spectral-v1" in color_ids
    assert "portra-160-spectral-v1" in color_ids
    assert "portra-800-spectral-v1" in color_ids
    assert "ektachrome-100-spectral-v1" in color_ids
    assert "provia-100f-spectral-v1" in color_ids
    assert "velvia-50-spectral-v1" in color_ids
    assert not (bw_ids & color_ids)

    bw_papers = list_paper_profiles(chemistry_mode="bw")
    color_papers = list_paper_profiles(chemistry_mode="color")
    assert any(p.stem.startswith("mg-") or "fiber" in p.stem for p in bw_papers)
    assert any("ra4" in p.stem for p in color_papers)


def test_named_color_profiles_use_brand_labels():
    portra = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    e100 = load_film_profile(ROOT / "profiles" / "films" / "ektachrome-100-spectral-v1.json")
    assert portra.name == "Kodak Portra 400"
    assert "Kodak" in str(portra.raw.get("manufacturer", ""))
    assert e100.name == "Kodak Ektachrome E100"
    doc = portra.raw["source"]["document"].lower()
    assert "approximate" in doc or "not a licensed" in doc


def test_c41_develop_has_mask_and_inverted_preview():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    assert chemistry_mode_for_film_type(profile.type) == "color"
    result = develop(
        dn,
        profile,
        developer_id="c41_standard",
        development_minutes=3.25,
        commit=False,
    )
    assert result.color_process == "c41"
    assert result.spectral_transmittance is not None
    assert result.spectral_transmittance.shape[-1] == N_WAVELENGTHS
    assert result.positive_preview.ndim == 3 and result.positive_preview.shape[-1] == 3
    # Dye concentrations include orange mask → not near-zero on mid grey.
    assert float(result.dye_concentrations.mean()) > 0.05
    # Light-table negative is orange-masked (R > G); scan invert is distinct.
    lightbox = color_negative_lightbox_preview(result.spectral_transmittance)
    lb_mean = lightbox.reshape(-1, 3).mean(0)
    pos_mean = result.positive_preview.reshape(-1, 3).mean(0)
    assert float(lb_mean[0]) > float(lb_mean[1])
    assert not np.allclose(lightbox, result.positive_preview, atol=1e-3)
    # Scan invert should not keep the strong orange R≫G bias.
    assert float(pos_mean[0] / max(float(pos_mean[1]), 1e-6)) < 1.35


def test_ra4_live_print_is_not_orange_mask():
    dn = ingest_path(None)
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    developed = develop(dn, film, developer_id="c41_standard", development_minutes=3.25, commit=False)
    dn.metadata.setdefault("print", {}).update({"cc_cyan": 0, "cc_magenta": 0, "cc_yellow": 0})
    printed = print_negative(
        developed.spectral_transmittance, dn, paper, base_exposure_seconds=8.0, commit=False
    )
    lightbox = color_negative_lightbox_preview(developed.spectral_transmittance)
    print_mean = printed.preview.reshape(-1, 3).mean(0)
    lb_mean = lightbox.reshape(-1, 3).mean(0)
    # Print should be brighter/neutraler than the orange light-table negative.
    assert float(print_mean.mean()) > 0.25
    assert float(print_mean[0]) < float(lb_mean[0]) * 1.15
    # Channels roughly balanced after dichroic stand-in (not strong cyan sludge).
    assert abs(float(print_mean[0] - print_mean[1])) < 0.12
    assert abs(float(print_mean[1] - print_mean[2])) < 0.12


def test_c41_push_pull_chemistries_resolve():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    std = develop(dn, profile, developer_id="c41_standard", development_minutes=3.25, commit=False)
    push = develop(dn, profile, developer_id="c41_push1", development_minutes=4.25, commit=False)
    # Push should shift densities / contrast relative to standard.
    assert not np.allclose(std.density, push.density, atol=1e-4)


def test_ra4_cc_filtration_moves_channels():
    dn = ingest_path(None)
    film = load_film_profile(ROOT / "profiles" / "films" / "portra-400-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    developed = develop(dn, film, developer_id="c41_standard", development_minutes=3.25, commit=False)
    t = developed.spectral_transmittance

    dn.metadata.setdefault("print", {}).update({"cc_cyan": 0, "cc_magenta": 0, "cc_yellow": 0})
    base = print_negative(t, dn, paper, base_exposure_seconds=8.0, commit=False)

    dn.metadata["print"].update({"cc_cyan": 0, "cc_magenta": 40, "cc_yellow": 0})
    tinted = print_negative(t, dn, paper, base_exposure_seconds=8.0, commit=False)
    # Magenta filtration should change the print (not identical).
    assert not np.allclose(base.preview, tinted.preview, atol=1e-4)


def test_e6_positive_finish():
    dn = ingest_path(None)
    film = load_film_profile(ROOT / "profiles" / "films" / "ektachrome-100-spectral-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "ra4-glossy-v1.json")
    developed = develop(dn, film, developer_id="e6_standard", development_minutes=6.0, commit=True)
    assert developed.color_process == "e6"
    assert dn.metadata["development"]["process"] == "e6"
    finished = print_negative(
        developed.spectral_transmittance, dn, paper, base_exposure_seconds=8.0, commit=True
    )
    assert finished.preview.ndim == 3
    assert dn.metadata["print"]["process"] == "e6_finish"


def test_bw_path_untouched_by_color_profiles():
    dn = ingest_path(None)
    film = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, film, developer_id="standard", commit=False)
    assert developed.color_process is None
    assert developed.spectral_transmittance is None
    assert developed.transmittance.ndim == 2
    printed = print_negative(
        developed.transmittance, dn, paper, base_exposure_seconds=8.0, grade=2.5, commit=False
    )
    assert printed.preview.ndim in (2, 3)
