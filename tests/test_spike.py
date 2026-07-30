"""Smoke tests for the darkroom pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_negative.curves import load_film_profile, modify_curve
from digital_negative.development import develop, linear_to_relative_log_exposure
from digital_negative.ingest import create_synthetic_scene, ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import run_darkroom_pipeline, run_spike_pipeline
from digital_negative.print_engine import print_negative


def test_hp5_profile_loads_and_interpolates():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    assert profile.id == "hp5-plus-v1"
    dens = profile.density_from_log_exposure(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert dens[0] < dens[1] < dens[2]
    assert dens[0] > profile.base_plus_fog


def test_fp4_is_finer_grain_than_hp5():
    hp5 = json.loads((ROOT / "profiles" / "films" / "hp5-plus-v1.json").read_text())
    fp4 = json.loads((ROOT / "profiles" / "films" / "fp4-plus-v1.json").read_text())
    assert fp4["grain_scale"] < hp5["grain_scale"]
    assert fp4["iso"] < hp5["iso"]


def test_delta_finer_than_fp4():
    fp4 = json.loads((ROOT / "profiles" / "films" / "fp4-plus-v1.json").read_text())
    delta = json.loads((ROOT / "profiles" / "films" / "delta-100-v1.json").read_text())
    assert delta["grain_scale"] < fp4["grain_scale"]


def test_tri_x_profile_loads_and_is_iso_400():
    path = ROOT / "profiles" / "films" / "tri-x-400-v1.json"
    data = json.loads(path.read_text())
    assert data["id"] == "tri-x-400-v1"
    assert data["iso"] == 400
    assert "F-4017" in data["source"]["document"]
    profile = load_film_profile(path)
    dens = profile.density_from_log_exposure(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    assert dens[0] < dens[1] < dens[2]
    assert dens[0] > profile.base_plus_fog
    hp5 = json.loads((ROOT / "profiles" / "films" / "hp5-plus-v1.json").read_text())
    assert data["grain_scale"] >= hp5["grain_scale"]


def test_film_chemistries_have_named_developers_and_minutes():
    from digital_negative.chemistry import (
        default_chemistry_id,
        minutes_to_relative,
        chemistries_map,
    )

    for film_name in ("tri-x-400-v1", "hp5-plus-v1", "fp4-plus-v1", "delta-100-v1"):
        profile = load_film_profile(ROOT / "profiles" / "films" / f"{film_name}.json")
        chems = chemistries_map(profile)
        assert chems, f"{film_name} missing chemistries"
        base_id = default_chemistry_id(profile)
        assert chems[base_id].get("is_base") or base_id in chems
        chem = chems[base_id]
        assert chem["normal_minutes"] > 0
        assert minutes_to_relative(chem, chem["normal_minutes"]) == pytest.approx(1.0, abs=0.02)
        longer = minutes_to_relative(chem, chem["normal_minutes"] * 1.5)
        assert longer > 1.05


def test_tri_x_d76_ci_push_increases_relative():
    from digital_negative.chemistry import get_chemistry, minutes_to_relative

    profile = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    chem = get_chemistry(profile, "d76")
    assert chem is not None
    normal = float(chem["normal_minutes"])
    assert minutes_to_relative(chem, normal) == pytest.approx(1.0, abs=0.02)
    assert minutes_to_relative(chem, 10.0) > minutes_to_relative(chem, normal)


def test_develop_accepts_minutes():
    from digital_negative.chemistry import default_chemistry_id, get_chemistry

    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    chem_id = default_chemistry_id(profile)
    chem = get_chemistry(profile, chem_id)
    result = develop(
        dn,
        profile,
        developer_id=chem_id,
        development_minutes=float(chem["normal_minutes"]),
        commit=False,
    )
    assert result.density.mean() > 0
    assert dn.metadata["development"]["development_minutes"] == pytest.approx(
        float(chem["normal_minutes"])
    )
    assert dn.metadata["development"]["relative_time"] == pytest.approx(1.0, abs=0.05)


def test_tri_x_d76_curve_family_interpolates_between_published_times():
    from digital_negative.curves import interpolate_curve_family, modify_curve
    from digital_negative.chemistry import get_chemistry

    profile = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    chem = get_chemistry(profile, "d76")
    assert chem is not None and len(chem["curve_family"]) >= 4
    log_e, dens6, fog6, meta6 = interpolate_curve_family(chem["curve_family"], 6.0)
    _, dens8, _, meta8 = interpolate_curve_family(chem["curve_family"], 8.0)
    _, dens7, _, meta7 = interpolate_curve_family(chem["curve_family"], 7.0)
    assert meta6["family_mode"] == "exact"
    assert meta8["family_mode"] == "exact"
    assert meta7["family_mode"] == "interpolated"
    # Mid/high tones should rise with time
    mid = int(0.65 * (len(dens6) - 1))
    assert dens8[mid] > dens6[mid]
    assert dens6[mid] < dens7[mid] < dens8[mid]

    worked = modify_curve(
        profile, developer_id="d76", development_minutes=7.0, contrast_modifier=0.0
    )
    assert worked.raw["_last_curve_meta"]["curve_source"] == "family"
    # Longer family time → denser highlights than shorter
    short = modify_curve(profile, developer_id="d76", development_minutes=6.0)
    long = modify_curve(profile, developer_id="d76", development_minutes=12.0)
    probe = np.array([3.5])
    assert float(long.density_from_log_exposure(probe)[0]) > float(
        short.density_from_log_exposure(probe)[0]
    )


def test_tri_x_tmax_uses_curve_family_not_morph():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "tri-x-400-v1.json")
    develop(
        dn,
        profile,
        developer_id="tmax",
        development_minutes=7.0,
        commit=False,
    )
    assert dn.metadata["development"]["curve_source"] == "family"


def test_ilford_without_family_still_morphs():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    develop(
        dn,
        profile,
        developer_id="ilfotec_hc_1_31",
        development_minutes=6.5,
        commit=False,
    )
    assert dn.metadata["development"]["curve_source"] == "morph"


def test_push_increases_midtone_density():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    normal = modify_curve(profile, relative_time=1.0)
    pushed = modify_curve(profile, relative_time=1.6)
    mid = np.array([2.3])
    assert float(pushed.density_from_log_exposure(mid)[0]) > float(
        normal.density_from_log_exposure(mid)[0]
    )


def test_high_energy_raises_contrast_vs_standard():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    std = modify_curve(profile, developer_id="standard")
    energy = modify_curve(profile, developer_id="high_energy")
    lo = np.array([1.6])
    hi = np.array([3.0])
    std_span = float(std.density_from_log_exposure(hi)[0] - std.density_from_log_exposure(lo)[0])
    energy_span = float(
        energy.density_from_log_exposure(hi)[0] - energy.density_from_log_exposure(lo)[0]
    )
    assert energy_span > std_span


def test_ingest_is_linear_xyz():
    dn = ingest_path(None)
    assert dn.metadata["ingest"]["encoding"] == "linear"
    assert dn.metadata["ingest"]["working_space"] == "CIE_XYZ"
    # Y channel is luminance for XYZ payloads
    assert np.allclose(dn.to_luminance(), dn.image[..., 1])


def test_ingest_synthetic_and_develop():
    dn = ingest_path(None)
    assert dn.image.ndim == 3
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    result = develop(dn, profile, grain_strength=0.5)
    assert result.density.shape == dn.to_luminance().shape
    assert 0.05 < float(result.density.mean()) < 2.5


def test_log_exposure_mapping_places_midtones():
    scene = create_synthetic_scene(128, 96)
    luma = scene[..., 1]  # CIE Y
    log_e = linear_to_relative_log_exposure(luma, mid_log_e=2.2)
    med = float(np.median(log_e))
    assert 1.8 < med < 2.6


def test_print_stage_darkens_with_more_exposure():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "fp4-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, profile, grain_strength=0.0)
    soft = print_negative(developed.transmittance, dn, paper, overall_exposure=-0.7, grade=2.0)
    hard = print_negative(developed.transmittance, dn, paper, overall_exposure=0.7, grade=2.0)
    assert float(hard.preview.mean()) < float(soft.preview.mean())


def test_hard_grade_increases_contrast_vs_soft():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, profile, grain_strength=0.0)
    soft = print_negative(developed.transmittance, dn, paper, overall_exposure=0.0, grade=0.0)
    hard = print_negative(developed.transmittance, dn, paper, overall_exposure=0.0, grade=5.0)
    soft_span = float(np.percentile(soft.preview, 95) - np.percentile(soft.preview, 5))
    hard_span = float(np.percentile(hard.preview, 95) - np.percentile(hard.preview, 5))
    assert hard_span > soft_span


def test_warmtone_paper_loads():
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-warmtone-v1.json")
    assert paper.id == "mg-warmtone"
    assert paper.dmax < 2.05


def test_fiber_paper_has_deeper_dmax():
    std = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    fiber = load_paper_profile(ROOT / "profiles" / "papers" / "fiber-glossy-v1.json")
    assert fiber.dmax > std.dmax


def test_hard_filter_is_slower_than_soft():
    """Without timer compensation, grade 5 should print lighter than grade 0."""
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, profile, grain_strength=0.0, process_variation=0.0)
    soft = print_negative(developed.transmittance, dn, paper, overall_exposure=0.0, grade=0.0)
    hard = print_negative(developed.transmittance, dn, paper, overall_exposure=0.0, grade=5.0)
    assert float(soft.preview.mean()) < float(hard.preview.mean())
    assert dn.metadata["print"]["filtration"]["values"]["filter_speed"] < 1.0


def test_push_builds_more_highlight_density_than_toe():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    normal = modify_curve(profile, relative_time=1.0)
    pushed = modify_curve(profile, relative_time=1.7)
    toe = np.array([1.3])
    hi = np.array([3.2])
    toe_gain = float(pushed.density_from_log_exposure(toe)[0] - normal.density_from_log_exposure(toe)[0])
    hi_gain = float(pushed.density_from_log_exposure(hi)[0] - normal.density_from_log_exposure(hi)[0])
    assert hi_gain > toe_gain


def test_full_pipeline_writes_print_artifacts(tmp_path: Path):
    artifacts = run_darkroom_pipeline(
        output_dir=tmp_path,
        film_id="delta-100-v1",
        paper_id="mg-warmtone",
        print_grade=3.0,
        grain_strength=0.8,
    )
    assert artifacts.comparison.exists()
    assert artifacts.print_preview is not None and artifacts.print_preview.exists()
    with artifacts.dn_json.open() as f:
        meta = json.load(f)
    assert meta["film_profile"]["id"] == "delta-100-v1"
    assert meta["ingest"]["working_space"] == "CIE_XYZ"
    assert meta["print"]["paper_id"] == "mg-warmtone"
    ops = {h.get("op") for h in meta["history"]}
    assert {"develop", "print"} <= ops


def test_spike_alias_skips_print(tmp_path: Path):
    artifacts = run_spike_pipeline(output_dir=tmp_path)
    assert artifacts.print_result is None
    assert artifacts.developed_preview.exists()


def test_contrast_modifier_steepens_curve():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    flat = modify_curve(profile, contrast_modifier=-0.8)
    steep = modify_curve(profile, contrast_modifier=0.8)
    lo = np.array([1.7])
    hi = np.array([3.0])
    flat_span = float(flat.density_from_log_exposure(hi)[0] - flat.density_from_log_exposure(lo)[0])
    steep_span = float(
        steep.density_from_log_exposure(hi)[0] - steep.density_from_log_exposure(lo)[0]
    )
    assert steep_span > flat_span


def test_high_definition_is_leaner_grain_than_standard():
    from digital_negative.curves import DEVELOPER_STYLES

    assert DEVELOPER_STYLES["high_definition"]["grain_bias"] < DEVELOPER_STYLES["standard"]["grain_bias"]
    assert DEVELOPER_STYLES["high_definition"]["density_bias"] < DEVELOPER_STYLES["standard"]["density_bias"]


def test_pull_reduces_highlight_density():
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    normal = modify_curve(profile, relative_time=1.0)
    pulled = modify_curve(profile, relative_time=0.7)
    hi = np.array([3.2])
    assert float(pulled.density_from_log_exposure(hi)[0]) < float(
        normal.density_from_log_exposure(hi)[0]
    )


def test_preview_develop_does_not_commit_history():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    before = len(dn.metadata.get("history", []))
    develop(dn, profile, grain_strength=0.2, process_variation=0.0, commit=False)
    assert len(dn.metadata.get("history", [])) == before
    assert "development" not in dn.metadata.get("ui_state", {}).get("committed_stages", [])


def test_preview_print_does_not_commit_history():
    dn = ingest_path(None)
    profile = load_film_profile(ROOT / "profiles" / "films" / "hp5-plus-v1.json")
    paper = load_paper_profile(ROOT / "profiles" / "papers" / "mg-standard-v1.json")
    developed = develop(dn, profile, grain_strength=0.0, process_variation=0.0, commit=True)
    before = len(dn.metadata.get("history", []))
    print_negative(developed.transmittance, dn, paper, commit=False)
    assert len(dn.metadata.get("history", [])) == before


def test_original_photo_preview_synthetic_is_rgb():
    from digital_negative.display import original_photo_preview

    dn = ingest_path(None)
    rgb = original_photo_preview(None, dn_image=dn.image)
    assert rgb.ndim == 3 and rgb.shape[-1] == 3
    assert rgb.dtype == np.uint8


def test_rotate_image_clockwise_swaps_axes():
    from digital_negative.display import rotate_image

    img = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    rotated = rotate_image(img, 1)
    assert rotated.shape == (4, 3, 2)
    assert np.array_equal(rotated, np.rot90(img, k=-1))


def test_resolve_input_prefers_upload_over_sample():
    sys.path.insert(0, str(ROOT / "scripts"))
    # Import after path setup — module lives as scripts/run_darkroom_ui.py
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_darkroom_ui", ROOT / "scripts" / "run_darkroom_ui.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._resolve_input("/tmp/mine.jpg", "/tmp/sample.nef") == "/tmp/mine.jpg"
    assert mod._resolve_input(None, "/tmp/sample.nef") == "/tmp/sample.nef"
    assert mod._resolve_input(None, None) is None
    assert mod._resolve_input(None, "") is None


def test_heif_ingest_decodes():
    pytest.importorskip("pillow_heif")
    from pillow_heif import register_heif_opener
    from PIL import Image

    register_heif_opener()
    path = ROOT / "output" / "_test_heif.heic"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((40, 60, 3), 140, dtype=np.uint8)).save(path)
    dn = ingest_path(path)
    assert dn.image.ndim == 3
    assert dn.image.shape[2] == 3
    assert float(dn.image.mean()) > 0
    assert "HEIF" in dn.metadata["ingest"]["notes"] or "HEIC" in dn.metadata["ingest"]["notes"]
    path.unlink(missing_ok=True)
