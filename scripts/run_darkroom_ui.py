#!/usr/bin/env python3
"""Darkroom UI: large commit-accurate live preview beside always-visible controls."""

from __future__ import annotations

import base64
import copy
import html as html_lib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from gradio import SelectData
import numpy as np

from digital_negative.auto_crop import (
    RULE_CHOICES as AUTO_CROP_RULE_CHOICES,
    RULE_LABELS as AUTO_CROP_RULE_LABELS,
    estimate_straighten_degrees,
    format_crop_rect,
    parse_aspect_ratio,
    suggest_crop_box,
)
from digital_negative.analysis import (
    apply_clipping_overlay,
    build_curve_report,
    curve_summary_markdown,
    render_curve_plot,
    render_print_histogram,
    spot_at,
    spot_markdown,
    suggest_tone_fit,
)
from digital_negative.recipes import build_recipe, load_recipe, save_recipe
from digital_negative.chemistry import (
    chemistry_choices,
    default_chemistry_id,
    get_chemistry,
    resolve_relative_time,
    time_slider_bounds,
)
from digital_negative.capture import FILTER_LABELS
from digital_negative.curves import load_film_profile
from digital_negative.color_development import color_negative_lightbox_preview
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative
from digital_negative.display import (
    apply_framing,
    linear_to_srgb,
    negative_lightbox_preview,
    original_photo_preview,
    rotate_image,
    straighten_image,
    to_u8_gray,
)
from digital_negative.dodge_burn import (
    CARD_PRESETS,
    REFERENCE_BASE_SECONDS,
    TICK_SECONDS,
    apply_exposure_tick,
    base_seconds_to_stops,
    ensure_accum,
    extract_tool_stamp,
    local_stops_from_state,
    parse_pointer,
    parse_pointer_state,
    relative_pass_stops,
    reset_local_work,
    resolve_tool_stamp,
    stamp_to_png_data_url,
    tool_workshop_canvas,
)
from digital_negative.ingest import ingest_path
from digital_negative.papers import load_paper_profile
from digital_negative.pipeline import list_film_profiles, list_paper_profiles
from digital_negative.print_engine import TONE_LABELS, print_negative

# Match commit look as closely as practical while staying interactive.
LIVE_MAX_SIDE = 2000
DRAG_MAX_SIDE = 1280  # high enough for critical judgment while dragging
INSPECT_MAX_SIDE = 3600  # high-res for zoom / inspect panel
REF_MAX_SIDE = 420
CROP_STAGE_MAX_SIDE = 1400

CROP_RATIO_CHOICES = [
    ("Free", "free"),
    ("Original", "original"),
    ("1:1", "1:1"),
    ("3:2", "3:2"),
    ("2:3", "2:3"),
    ("4:3", "4:3"),
    ("3:4", "3:4"),
    ("5:4", "5:4"),
    ("4:5", "4:5"),
    ("16:9", "16:9"),
    ("9:16", "9:16"),
]
DEFAULT_CROP_RECT = "0.00000,0.00000,1.00000,1.00000"

def _film_choice_tuple(path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (f"{data['name']} (ISO {data['iso']})", data["id"])


def _paper_choice_tuple(path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data["name"], data["id"])


FILM_CHOICES_BW = [_film_choice_tuple(p) for p in list_film_profiles(chemistry_mode="bw")]
FILM_CHOICES_COLOR = [_film_choice_tuple(p) for p in list_film_profiles(chemistry_mode="color")]
FILM_CHOICES = list(FILM_CHOICES_BW)  # default chemistry mode = B&W

PAPER_CHOICES_BW = [_paper_choice_tuple(p) for p in list_paper_profiles(chemistry_mode="bw")]
PAPER_CHOICES_COLOR = [_paper_choice_tuple(p) for p in list_paper_profiles(chemistry_mode="color")]
PAPER_CHOICES = list(PAPER_CHOICES_BW)

CHEMISTRY_MODE_LABELS = [
    ("Black & White Chemistry", "bw"),
    ("Color Chemistry", "color"),
]

# Filled after helper defs below (film-specific chemistry dropdown + minutes).
_INIT_DEV_CHOICES: list = []
_INIT_DEV_ID = "standard"
_INIT_TMIN, _INIT_TMAX, _INIT_TNORM = 4.0, 16.0, 8.0

SAMPLE_DIR = ROOT / "samples" / "raws"
SAMPLE_CHOICES = [("Synthetic test scene (no file)", "")]
if SAMPLE_DIR.exists():
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}:
            SAMPLE_CHOICES.append((path.name, str(path)))

UI_CSS = """
/* ——— Darkroom theme: warm-neutral dark chrome, copper accent ——— */
:root, .gradio-container {
  --dr-bg-app: #16161a !important;
  --dr-bg-panel: #1d1d21 !important;
  --dr-bg-elevated: #262629 !important;
  --dr-bg-hover: #2d2d31 !important;
  --dr-border: rgba(255, 255, 255, 0.09) !important;
  --dr-border-strong: rgba(255, 255, 255, 0.17) !important;
  --dr-text: #eae6df !important;
  --dr-text-dim: #a8a49b !important;
  --dr-text-faint: #757168 !important;
  --dr-accent: #e0954f !important;
  --dr-accent-strong: #f0ab6c !important;
  --dr-accent-soft: rgba(224, 149, 79, 0.16) !important;
  --dr-accent-border: rgba(224, 149, 79, 0.5) !important;
  --dr-accent-contrast: #20140a !important;
  --dr-radius: 9px !important;
  --dr-font: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Helvetica, Arial, sans-serif !important;
  /* One type scale for both side panels so labels stay on a single line. */
  --dr-fs-label: 0.62rem !important;
  --dr-fs-control: 0.63rem !important;
  --dr-fs-note: 0.58rem !important;
  --dr-fs-title: 0.56rem !important;
  --dr-fs-tiny: 0.52rem !important;

  /* Re-point Gradio's own theme variables at the darkroom palette so
     sliders, primary buttons, and focus rings match without touching
     every component individually. */
  --color-accent: var(--dr-accent) !important;
  --color-accent-soft: var(--dr-accent-soft) !important;
  --border-color-accent: var(--dr-accent-border) !important;
  --border-color-accent-subdued: var(--dr-accent-soft) !important;
  --slider-color: var(--dr-accent) !important;
  --button-primary-background-fill: var(--dr-accent) !important;
  --button-primary-background-fill-hover: var(--dr-accent-strong) !important;
  --button-primary-border-color: var(--dr-accent) !important;
  --button-primary-border-color-hover: var(--dr-accent-strong) !important;
  --button-primary-text-color: var(--dr-accent-contrast) !important;
  --button-primary-text-color-hover: var(--dr-accent-contrast) !important;
  --button-secondary-background-fill: var(--dr-bg-elevated) !important;
  --button-secondary-background-fill-hover: var(--dr-bg-hover) !important;
  --button-secondary-border-color: var(--dr-border-strong) !important;
  --button-secondary-border-color-hover: var(--dr-accent-border) !important;
  --button-secondary-text-color: var(--dr-text) !important;
  --button-secondary-text-color-hover: var(--dr-text) !important;
  --checkbox-label-background-fill: var(--dr-bg-elevated) !important;
  --checkbox-label-background-fill-hover: var(--dr-bg-hover) !important;
  --checkbox-label-background-fill-selected: var(--dr-accent-soft) !important;
  --checkbox-label-border-color: var(--dr-border) !important;
  --checkbox-label-border-color-selected: var(--dr-accent-border) !important;
  --checkbox-label-text-color: var(--dr-text-dim) !important;
  --checkbox-label-text-color-selected: var(--dr-accent-strong) !important;
  --checkbox-background-color: var(--dr-bg-elevated) !important;
  --checkbox-background-color-selected: var(--dr-accent) !important;
  --checkbox-border-color: var(--dr-border-strong) !important;
  --checkbox-border-color-selected: var(--dr-accent) !important;
  --body-background-fill: var(--dr-bg-app) !important;
  --background-fill-primary: var(--dr-bg-panel) !important;
  --background-fill-secondary: var(--dr-bg-elevated) !important;
  --block-background-fill: var(--dr-bg-panel) !important;
  --block-border-color: var(--dr-border) !important;
  --body-text-color: var(--dr-text) !important;
  --body-text-color-subdued: var(--dr-text-dim) !important;
  --border-color-primary: var(--dr-border) !important;
  --input-background-fill: var(--dr-bg-elevated) !important;
  --input-border-color: var(--dr-border) !important;
  --neutral-50: var(--dr-bg-elevated) !important;
}
html, body {
  height: 100% !important;
  overflow: hidden !important;
  margin: 0 !important;
  background: var(--dr-bg-app) !important;
}
.gradio-container {
  max-width: 100% !important;
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
  padding: 0 2px 0 !important;
  box-sizing: border-box !important;
  display: flex !important;
  flex-direction: column !important;
  background: var(--dr-bg-app) !important;
  font-family: var(--dr-font) !important;
  color: var(--dr-text) !important;
}
/* Gradio's shell adds page padding, side margins, a max-width and a 16px
   column gap. Strip it all so the workspace is genuinely full-bleed. */
.gradio-container > .main,
.gradio-container .main.fillable,
.gradio-container .wrap.svelte-1jdub1s,
.gradio-container .contain {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
  max-width: 100% !important;
  width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
  overflow: hidden !important;
}
.gradio-container .contain > .column,
.gradio-container .main > .wrap > .contain > div {
  gap: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  height: 100% !important;
  min-height: 0 !important;
}
/* Empty 2px form stub Gradio emits above the workspace. */
.gradio-container .contain > .column > .form:empty { display: none !important; }
/* The photo is the point of the app — the title bar is not worth the pixels. */
#app_header { display: none !important; }
/* Gradio's footer would otherwise overlap the bottom of the workspace. */
footer, .gradio-container footer {
  position: fixed !important;
  bottom: 0 !important;
  left: 0 !important;
  right: 0 !important;
  height: 18px !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  font-size: 0.62rem !important;
  opacity: 0.35 !important;
  background: var(--dr-bg-app) !important;
  z-index: 5 !important;
}

/* Fixed non-scrolling workspace — only the slim fixed footer is reserved.
   min-height keeps a short drawer (e.g. Frame) from collapsing the print. */
#main_workspace {
  flex: 1 1 auto !important;
  min-height: calc(100vh - 20px) !important;
  height: calc(100vh - 20px) !important;
  max-height: calc(100vh - 20px) !important;
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  gap: 0 !important;
  overflow: hidden !important;
}
#main_workspace > div {
  min-width: 0 !important;
  min-height: 0 !important;
  height: 100% !important;
}

/* Icon rail — darktable-style: flat, quiet, accent only on the active stage */
#icon_rail {
  flex: 0 0 42px !important;
  width: 42px !important;
  max-width: 42px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 1px !important;
  padding: 3px 3px 4px !important;
  box-sizing: border-box !important;
  border-right: 1px solid var(--dr-border) !important;
  background: var(--dr-bg-app) !important;
  z-index: 40 !important;
  overflow: hidden !important;
}
#icon_rail button {
  min-height: 38px !important;
  height: 38px !important;
  width: 100% !important;
  padding: 3px 1px 2px !important;
  font-size: 0.48rem !important;
  line-height: 1 !important;
  white-space: pre-line !important;
  border-radius: 6px !important;
  letter-spacing: 0.01em;
  color: var(--dr-text-dim) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  transition: background 0.12s ease, color 0.12s ease !important;
}
#icon_rail button:hover {
  background: var(--dr-bg-hover) !important;
  color: var(--dr-text) !important;
}
#icon_rail button .rail-glyph,
#icon_rail .rail-glyph {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1px;
}
#icon_rail button .rail-svg {
  width: 16px;
  height: 16px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
  display: block;
}
#icon_rail button .rail-label {
  display: block;
  font-size: 0.48rem;
  line-height: 1;
}
#icon_rail button.rail-active {
  color: var(--dr-accent-strong) !important;
  background: var(--dr-accent-soft) !important;
  border-color: var(--dr-accent-border) !important;
}
#icon_rail .rail-spacer { flex: 1 1 auto !important; min-height: 8px !important; }

/* Drawer host — one panel visible; compressed.
   Keep every nested block inside the rail width — Gradio slider min/max
   rows and wide number boxes were forcing horizontal scrollbars. */
#drawer_host {
  flex: 0 0 188px !important;
  width: 188px !important;
  max-width: 188px !important;
  min-width: 0 !important;
  height: 100% !important;
  flex-wrap: nowrap !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  /* Extra bottom pad so Commit Print/Develop aren't clipped at the scroll end
     (Color Print is tall: CC row + split-grade + strips). */
  padding: 4px 6px 56px !important;
  box-sizing: border-box !important;
  border-right: 1px solid var(--dr-border) !important;
  background: var(--dr-bg-panel) !important;
  z-index: 35 !important;
  overscroll-behavior: contain !important;
  scrollbar-gutter: stable !important;
  transition: flex-basis 0.18s ease, width 0.18s ease, max-width 0.18s ease, padding 0.18s ease, opacity 0.15s ease !important;
}
/* Keep stage commit actions reachable while scrolling a long Print drawer. */
#drawer_host #print_commit_row,
#drawer_host #develop_commit_row {
  position: sticky !important;
  bottom: 0 !important;
  z-index: 6 !important;
  margin: 6px 0 0 !important;
  padding: 6px 0 4px !important;
  background: linear-gradient(
    to bottom,
    rgba(18, 18, 20, 0) 0%,
    var(--dr-bg-panel) 28%,
    var(--dr-bg-panel) 100%
  ) !important;
  border-top: 1px solid var(--dr-border) !important;
  gap: 4px !important;
}
#drawer_host #print_commit_row button,
#drawer_host #develop_commit_row button {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}
#drawer_host,
#drawer_host .drawer-panel,
#drawer_host .gr-accordion,
#drawer_host .block,
#drawer_host .form,
#drawer_host .wrap,
#drawer_host .container,
#drawer_host .styler,
#drawer_host > div {
  max-width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
}
#drawer_host .block,
#drawer_host .form,
#drawer_host .wrap,
#drawer_host .container,
#drawer_host .head,
#drawer_host [data-testid="slider"],
#drawer_host .slider-container {
  overflow-x: hidden !important;
  width: 100% !important;
  max-width: 100% !important;
}
/* Hide the min/max captions under drawer sliders — in a ~180px rail they
   overflow ("25 … 6400", "0.001 … 120") and Gradio paints a horizontal bar. */
#drawer_host .min_value,
#drawer_host .max_value,
#drawer_host .min-val,
#drawer_host .max-val,
#drawer_host span.min,
#drawer_host span.max {
  display: none !important;
}
#drawer_host input[type="range"] {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
#drawer_host .head {
  display: flex !important;
  align-items: center !important;
  gap: 2px !important;
  width: 100% !important;
}
#drawer_host .head input[type="number"] {
  width: 40px !important;
  min-width: 0 !important;
  max-width: 44px !important;
  flex: 0 0 40px !important;
  box-sizing: border-box !important;
}
/* Camera roll tab — server-rendered HTML list (✕ is a real button).
   Thumbs keep a fixed height (flex-shrink: 0); the list scrolls instead of
   compressing every frame to fit the drawer. */
#drawer_roll #camera_roll,
#drawer_roll #camera_roll > .wrap,
#drawer_roll #camera_roll > .html-container,
#drawer_roll #camera_roll .prose {
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  overflow: visible !important;
  height: auto !important;
  max-height: none !important;
}
#drawer_roll #camera_roll .roll-list {
  display: flex !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  justify-content: flex-start !important;
  gap: 6px !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  overscroll-behavior: contain !important;
  /* Fill the drawer column; scroll when frames exceed the viewport. */
  max-height: min(72vh, calc(100vh - 120px)) !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 0 2px 4px 0 !important;
  margin: 0 !important;
  flex: 0 1 auto !important;
}
#drawer_roll #camera_roll .roll-empty {
  color: var(--dr-text-dim) !important;
  font-size: var(--dr-fs-label) !important;
  padding: 8px 2px !important;
  flex: 0 0 auto !important;
}
#drawer_roll #camera_roll .roll-item {
  position: relative !important;
  /* Critical: default flex-shrink:1 was squashing 88px thumbs into strips. */
  flex: 0 0 88px !important;
  flex-grow: 0 !important;
  flex-shrink: 0 !important;
  align-self: stretch !important;
  width: 100% !important;
  height: 88px !important;
  min-height: 88px !important;
  max-height: 88px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 6px !important;
  background: #0a0a0a !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
  cursor: pointer !important;
}
#drawer_roll #camera_roll .roll-item.is-active {
  border-color: var(--dr-accent) !important;
  box-shadow: inset 0 0 0 1px var(--dr-accent) !important;
}
#drawer_roll #camera_roll .roll-item img {
  display: block !important;
  width: 100% !important;
  height: 100% !important;
  min-height: 88px !important;
  object-fit: cover !important;
  object-position: center center !important;
  pointer-events: none !important;
  flex-shrink: 0 !important;
}
#drawer_roll #camera_roll .roll-cap {
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  margin: 0 !important;
  padding: 3px 5px !important;
  font-size: 0.62rem !important;
  line-height: 1.15 !important;
  color: var(--dr-text) !important;
  background: rgba(12, 12, 14, 0.86) !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  pointer-events: none !important;
}
#drawer_roll #camera_roll .roll-x {
  position: absolute !important;
  top: 5px !important;
  right: 5px !important;
  z-index: 6 !important;
  width: 22px !important;
  min-width: 22px !important;
  height: 22px !important;
  min-height: 22px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  border-radius: 5px !important;
  background: rgba(12, 12, 14, 0.82) !important;
  color: var(--dr-text) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  line-height: 20px !important;
  text-align: center !important;
  cursor: pointer !important;
  opacity: 0 !important;
  transition: opacity 0.12s ease, background 0.12s ease, color 0.12s ease !important;
}
#drawer_roll #camera_roll .roll-item:hover .roll-x {
  opacity: 1 !important;
}
#drawer_roll #camera_roll .roll-x:hover {
  background: rgba(160, 48, 36, 0.92) !important;
  border-color: transparent !important;
  color: #fff !important;
}
#drawer_roll #roll_meta {
  margin: 0 0 6px !important;
  font-size: var(--dr-fs-label) !important;
  color: var(--dr-text-dim) !important;
}
/* Park Gradio triggers off-screen but keep them mounted (not visible=False),
   so JS value writes + clicks still reach the backend. */
#roll_remove_index,
#roll_remove,
#roll_switch_index,
#roll_switch,
#roll_pending_index {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
}

/* Save-before-switch prompt — fixed overlay above the darkroom. */
#roll_save_modal {
  position: fixed !important;
  inset: 0 !important;
  z-index: 2000 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: rgba(0, 0, 0, 0.58) !important;
  padding: 16px !important;
  box-sizing: border-box !important;
}
#roll_save_modal.hidden,
#roll_save_modal[style*="display: none"],
#roll_save_modal:not(.show) {
  /* Gradio toggles visibility; keep our flex when shown. */
}
#roll_save_dialog {
  width: min(380px, 92vw) !important;
  margin: 0 !important;
  padding: 16px 16px 12px !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 10px !important;
  background: #161618 !important;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45) !important;
}
#roll_save_dialog .prose,
#roll_save_dialog .prose * {
  color: var(--dr-text) !important;
  font-size: 0.88rem !important;
  line-height: 1.35 !important;
}
#roll_save_actions {
  gap: 8px !important;
  margin-top: 12px !important;
}
#roll_save_actions button {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  font-size: 0.78rem !important;
}

/* Kill any leftover horizontal scrollbar chrome inside the drawer. */
#drawer_host *::-webkit-scrollbar:horizontal {
  height: 0 !important;
  display: none !important;
}
body.drawer-collapsed #drawer_host {
  flex-basis: 0 !important;
  width: 0 !important;
  max-width: 0 !important;
  padding: 0 !important;
  opacity: 0 !important;
  border-right-width: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
.drawer-panel { display: none !important; }
.drawer-panel.is-open { display: block !important; }
.drawer-panel .gr-accordion > .label-wrap {
  display: none !important; /* rail is the chrome */
}
/* Because that header is hidden, a collapsed accordion is a dead end: the
   drawer renders as an empty box with no way to reopen it. The ritual still
   sends open=False as stages lock, so pin drawer content open regardless. */
.drawer-panel [data-testid="accordion-content"] {
  display: block !important;
  overflow-x: hidden !important;
  overflow-y: visible !important;
  max-width: 100% !important;
  /* Room below the last control so sticky commit row + scroll end clear. */
  padding-bottom: 8px !important;
}
.drawer-panel .gr-accordion {
  margin: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
#drawer_host .block {
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
}
/* Slider/number rows: label above, control below, both tight.
   Labels wrapped to two and three lines in a 176px panel, so they're
   clamped to one line and ellipsised instead.
   Gradio nests a <span> inside the <label> that re-declares 14px, so the
   span has to be targeted too or the text renders full-size and clips. */
#drawer_host .label-wrap,
#drawer_host label,
#drawer_host label span,
#drawer_host .head span,
#drawer_host [data-testid="block-label"],
#drawer_host [data-testid="block-label"] span {
  margin: 0 !important;
  padding: 0 !important;
  font-size: var(--dr-fs-label) !important;
  line-height: 1.15 !important;
  color: var(--dr-text-dim) !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 100% !important;
}
/* Give the label the row: the number box + reset button were taking 82px
   of a 159px head, squeezing the label to 77px for ~120px of text. */
#drawer_host .head .tab-like-container,
#module_panel .head .tab-like-container {
  flex: 0 0 auto !important;
  gap: 0 !important;
}
#drawer_host .head input[type="number"],
#module_panel .head input[type="number"] {
  width: 38px !important;
  min-width: 38px !important;
  text-align: right !important;
}
#drawer_host .head label,
#module_panel .head label {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}
/* Radio/checkbox options must stay wrappable — they're multi-word choices,
   and the nowrap/ellipsis clamp above would otherwise truncate them. */
#drawer_host fieldset label,
#drawer_host fieldset label span,
#module_panel fieldset label,
#module_panel fieldset label span {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
/* Gradio's .prose sets its own 14px, which dwarfed the 10px controls and let
   the explanatory notes dominate both panels. Pin markdown to the note size. */
#drawer_host .prose,
#drawer_host .prose *,
#module_panel .prose,
#module_panel .prose *,
#ritual_status .prose,
#ritual_status .prose * {
  font-size: var(--dr-fs-note) !important;
  line-height: 1.3 !important;
}
#drawer_host .prose p,
#module_panel .prose p,
#ritual_status .prose p {
  margin: 0 0 3px 0 !important;
}
#drawer_host .prose :is(h1, h2, h3, h4),
#module_panel .prose :is(h1, h2, h3, h4) {
  font-size: var(--dr-fs-label) !important;
  margin: 2px 0 !important;
}
#drawer_host .form {
  gap: 0 !important;
  border: none !important;
  background: transparent !important;
}
#drawer_host button {
  min-height: 21px !important;
  height: 21px !important;
  font-size: var(--dr-fs-control) !important;
  padding: 1px 5px !important;
  border-radius: 5px !important;
}
#drawer_host input,
#drawer_host select,
#drawer_host .wrap-inner,
#drawer_host .secondary-wrap input {
  font-size: var(--dr-fs-control) !important;
  min-height: 19px !important;
  height: 19px !important;
  padding: 0 4px !important;
}
/* Slider track + its number box */
#drawer_host input[type="range"] { height: 12px !important; margin: 0 !important; }
#drawer_host .head { font-size: var(--dr-fs-control) !important; margin: 0 !important; }
#drawer_host .container > .wrap,
#drawer_host .block > .wrap { padding: 0 !important; }
#drawer_host .icon-button-wrapper,
#drawer_host .reset-button { transform: scale(0.8) !important; }
/* Keep the dropzone strictly inside its own box — with overflow visible it
   spilled past its bounds and swallowed clicks meant for Commit Ingest.
   The upload-in-progress state (spinner + file name + progress bar) needs
   more room than the idle dropzone, so allow it to grow to fit. */
#ingest_upload {
  overflow: hidden !important;
  min-height: 74px !important;
  max-height: 120px !important;
  contain: layout paint !important;
}
#ingest_upload .wrap,
#ingest_upload .upload-container,
#ingest_upload .center,
#ingest_upload [data-testid="file"] {
  overflow: hidden !important;
  min-height: 0 !important;
  max-height: 118px !important;
}
/* Uploading state: keep the file name on one line and let the progress bar
   and byte counter sit inside the box instead of being sliced. */
#ingest_upload .file-preview,
#ingest_upload .progress-bar-wrap,
#ingest_upload .progress-text,
#ingest_upload .file-name,
#ingest_upload .filename,
#ingest_upload td,
#ingest_upload .progress-bar {
  font-size: var(--dr-fs-tiny) !important;
  line-height: 1.25 !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
#ingest_upload .file-name,
#ingest_upload .filename,
#ingest_upload td {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
#ingest_upload table, #ingest_upload tbody, #ingest_upload tr {
  width: 100% !important;
  table-layout: fixed !important;
  display: block !important;
}
/* The uploaded-file preview ships 10px of padding on the table, the row and
   every cell — ~105px of box for two short lines, which overflowed. */
#ingest_upload .file-preview-holder {
  max-height: 100% !important;
  overflow: hidden !important;
}
#ingest_upload table.file-preview { padding: 1px 2px !important; margin: 0 !important; }
#ingest_upload tr.file { padding: 0 !important; }
#ingest_upload td.filename,
#ingest_upload td.download { padding: 0 3px !important; }
#ingest_upload td.filename span { display: inline !important; }
#ingest_upload .wrap {
  height: auto !important;
}
#ingest_upload .upload-container,
#ingest_upload .center {
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  padding: 4px !important;
  min-height: 0 !important;
  box-sizing: border-box !important;
  font-size: var(--dr-fs-note) !important;
  line-height: 1.15 !important;
}
/* Gradio's dropzone renders its caption at 16px above a 27px upload icon.
   Squeezed into the compact drawer the icon was laid out above the box
   (clipped) and the caption collided with the floating "Upload" label.
   Drop the icon and put the caption on the panel type scale. */
#ingest_upload .icon-wrap { display: none !important; }
#ingest_upload button.center {
  height: 100% !important;
  min-height: 0 !important;
  /* clear the floating "Upload" label sitting at the top of the box */
  padding: 13px 4px 3px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
  box-sizing: border-box !important;
}
#ingest_upload button.center,
#ingest_upload button.center *,
#ingest_upload .or {
  font-size: var(--dr-fs-note) !important;
  line-height: 1.2 !important;
  margin: 0 !important;
}
#ingest_upload button,
#ingest_upload .or,
#ingest_upload span,
#ingest_upload p {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
  max-height: none !important;
}
#drawer_host [role="listbox"],
#drawer_host ul.options {
  z-index: 9999 !important;
  max-height: min(40vh, 280px) !important;
  overflow-y: auto !important;
}
/* Stage + recipe readout: a transparent, collapsible float over the print,
   pinned under the mode pill instead of eating drawer space. */
#ritual_status {
  position: absolute !important;
  top: 30px !important;
  left: 10px !important;
  z-index: 7 !important;
  width: auto !important;
  max-width: min(300px, 42%) !important;
  margin: 0 !important;
  padding: 5px 22px 5px 9px !important;
  font-size: var(--dr-fs-note) !important;
  line-height: 1.3 !important;
  color: var(--dr-text-dim) !important;
  background: rgba(18, 18, 21, 0.62) !important;
  backdrop-filter: blur(4px) !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 7px !important;
  max-height: 44vh !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  transition: padding 0.12s ease, background 0.12s ease !important;
}
#ritual_status p { margin: 0 0 2px 0 !important; }
#ritual_status p:last-child { margin-bottom: 0 !important; }
#ritual_status strong { color: var(--dr-text) !important; }
/* Collapse toggle injected by the UI script. */
#ritual_status .status-toggle {
  position: absolute;
  top: 3px;
  right: 4px;
  width: 15px;
  height: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: var(--dr-text-faint);
  cursor: pointer;
  font-size: 0.6rem;
  line-height: 1;
  padding: 0;
}
#ritual_status .status-toggle:hover {
  background: var(--dr-bg-hover);
  color: var(--dr-text);
}
/* Collapsed: shrink to a small chip showing just the current stage. */
#ritual_status.status-collapsed {
  max-height: 22px !important;
  padding: 3px 22px 3px 9px !important;
  overflow: hidden !important;
}
#ritual_status.status-collapsed .prose > *:not(:first-child) { display: none !important; }
#ritual_status.status-collapsed .prose > *:first-child {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  margin: 0 !important;
}
#ritual_status:hover { background: rgba(18, 18, 21, 0.86) !important; }
#history_box {
  max-height: calc(100vh - 90px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  font-size: 0.8rem !important;
  line-height: 1.35 !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
}
#history_box *,
#history_box code {
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  white-space: pre-wrap !important;
}
/* Markdown code spans default to a near-white fill, which on this dark
   chrome turned every logged value into a glaring highlight. */
#history_box code,
#history_box pre,
#history_box kbd,
#history_box samp,
#drawer_host code,
#module_panel code {
  background: rgba(255, 255, 255, 0.07) !important;
  color: var(--dr-accent-strong) !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 4px !important;
  padding: 0 3px !important;
  font-size: 0.95em !important;
  box-shadow: none !important;
}
#history_box pre {
  padding: 3px 5px !important;
  background: rgba(255, 255, 255, 0.04) !important;
}
#history_box pre code {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
#history_box strong { color: var(--dr-text) !important; }
#history_box em { color: var(--dr-text-dim) !important; }
#history_box a { color: var(--dr-accent-strong) !important; }
/* Table-ish log rows and blockquotes were also light-filled. */
#history_box blockquote {
  border-left: 2px solid var(--dr-accent-border) !important;
  background: rgba(255, 255, 255, 0.03) !important;
  margin: 2px 0 !important;
  padding: 2px 6px !important;
}
#history_box mark {
  background: var(--dr-accent-soft) !important;
  color: var(--dr-text) !important;
}

/* Preview fills remaining viewport — image scales to the stage (object-fit: contain) */
#preview_col {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  min-height: calc(100vh - 22px) !important;
  height: calc(100vh - 22px) !important;
  display: flex !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  gap: 2px !important;
  padding: 2px 3px 2px !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
  z-index: 1 !important;
  position: relative !important;
}
/* The status pill floats over the print instead of reserving a strip of it. */
#db_size_readout {
  position: absolute !important;
  top: 5px !important;
  left: 10px !important;
  z-index: 6 !important;
  width: auto !important;
  min-width: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  pointer-events: none !important;
}
#db_wave_banner {
  flex: 0 0 auto !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}
#db_wave_banner:empty { display: none !important; }
#preview_col > .block:has(#live_preview),
#live_preview {
  flex: 1 1 0 !important;
  min-height: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
  overflow: hidden !important;
}
#live_preview > .wrap,
#live_preview .wrap,
#live_preview .image-frame,
#live_preview .image-container,
#live_preview [data-testid="image"],
#live_preview .image-container > div {
  width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  max-height: 100% !important;
  overflow: hidden !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: #0a0a0a !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: var(--dr-radius) !important;
  box-sizing: border-box !important;
}
#live_preview img,
#live_preview .image-container img,
#live_preview .image-frame img {
  max-width: 100% !important;
  max-height: 100% !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
}
/* Hide Gradio chrome that steals stage height / covers the print */
#live_preview .icon-wrap,
#live_preview .download,
#live_preview button.svelte-1w6vlo0,
#live_preview .image-button-row,
#live_preview .icon-button-wrapper,
#live_preview .top-panel,
#live_preview .icon-button-wrapper.top-panel {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  opacity: 0 !important;
  pointer-events: none !important;
  background: transparent !important;
  overflow: hidden !important;
}
/* Keep the print's own caption, but pin it top-right so it can't collide
   with the mode pill floating at top-left. */
#live_preview .label-wrap,
#live_preview [data-testid="block-label"] {
  position: absolute !important;
  top: 5px !important;
  right: 8px !important;
  left: auto !important;
  z-index: 2 !important;
  opacity: 0.5 !important;
  pointer-events: none !important;
  font-size: 0.62rem !important;
  line-height: 1 !important;
  padding: 2px 7px !important;
  border: none !important;
  border-radius: 999px !important;
  background: rgba(20, 20, 23, 0.72) !important;
  white-space: nowrap !important;
}
#live_preview [data-testid="block-label"] svg { display: none !important; }
/* Filmstrip — thin, fixed height, never allowed to push past the viewport */
#seq_strip {
  flex: 0 0 auto !important;
  flex-wrap: nowrap !important;
  justify-content: flex-start !important;
  gap: 5px !important;
  align-items: center !important;
  height: 46px !important;
  min-height: 46px !important;
  max-height: 46px !important;
  overflow: hidden !important;
  padding: 0 !important;
  margin: 0 !important;
}
#seq_strip .block {
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  /* Without an explicit height these blocks render 56px and spill out of the
     46px strip, clipping against the footer. */
  height: 44px !important;
  min-height: 44px !important;
  max-height: 44px !important;
  /* Stretched full-width they became 375px letterbox slivers of the frame;
     a fixed thumb width keeps the whole picture readable. */
  flex: 0 0 68px !important;
  width: 68px !important;
  min-width: 68px !important;
  max-width: 68px !important;
  align-self: center !important;
}
#seq_strip .image-container,
#seq_strip .image-frame,
#seq_strip [data-testid="image"],
#seq_strip .image-container > button {
  height: 44px !important;
  min-height: 44px !important;
  max-height: 44px !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 5px !important;
  background: #0a0a0a !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
}
#seq_strip .image-container > button { border: none !important; }
/* Hover buttons off the thumbs — they were eating the whole 44px, leaving
   the actual frame as a sliver. */
#seq_strip .label-wrap,
#seq_strip .icon-button-wrapper,
#seq_strip .icon-button,
#seq_strip .top-panel,
#seq_strip .icon-wrap,
#seq_strip .download,
#seq_strip .image-button-row,
#seq_strip [aria-label="Fullscreen"] {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
/* The stage name rides on the thumb and only shows on hover. */
#seq_strip [data-testid="block-label"] {
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  top: auto !important;
  z-index: 4 !important;
  margin: 0 !important;
  padding: 2px 0 !important;
  border: none !important;
  border-radius: 0 0 5px 5px !important;
  background: rgba(12, 12, 14, 0.86) !important;
  color: var(--dr-text) !important;
  font-size: 0.6rem !important;
  line-height: 1.1 !important;
  text-align: center !important;
  opacity: 0 !important;
  transition: opacity 0.12s ease !important;
  pointer-events: none !important;
  white-space: nowrap !important;
}
#seq_strip [data-testid="block-label"] svg { display: none !important; }
#seq_strip .block:hover [data-testid="block-label"] { opacity: 1 !important; }
#seq_strip .block { cursor: pointer !important; position: relative !important; }
#seq_strip .block:hover .image-container { border-color: var(--dr-accent) !important; }
#seq_strip img {
  max-height: 42px !important;
  max-width: 100% !important;
  width: auto !important;
  height: auto !important;
  /* contain, not cover — the point is to recognise the frame at a glance. */
  object-fit: contain !important;
  border-radius: 4px !important;
}
#seq_strip button {
  min-height: 22px !important;
  height: 22px !important;
  font-size: 0.62rem !important;
  padding: 1px 6px !important;
  border-radius: 5px !important;
}
/* The real download buttons are clicked by the popup, never shown. */
#dl_pkg_print, #dl_pkg_both, #dl_pkg_negative, #download_modes {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
#preview_tool, #active_drawer, #crop_rect, #db_pos, #curves_open, #spot_pos, #inspect_open {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
#inspect_preview { display: none !important; }

/* Gradio's internal .styler wrapper tints components with a translucent
   white overlay by default — flatten it everywhere in the dark chrome. */
.gradio-container .styler {
  background: transparent !important;
}

/* ——— Persistent module panel (darktable-style, right side) ——— */
#module_panel {
  flex: 0 0 190px !important;
  width: 190px !important;
  max-width: 190px !important;
  min-width: 0 !important;
  height: 100% !important;
  /* Gradio's own Column class defaults to flex-wrap: wrap — once open modules'
     combined height exceeded the panel, later ones wrapped into an invisible
     second column instead of the panel scrolling. Force a single column. */
  flex-wrap: nowrap !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  padding: 3px 6px 6px !important;
  box-sizing: border-box !important;
  border-left: 1px solid var(--dr-border) !important;
  background: var(--dr-bg-panel) !important;
  z-index: 35 !important;
  transition: flex-basis 0.16s ease, width 0.16s ease, padding 0.16s ease, opacity 0.14s ease !important;
}
body.module-collapsed #module_panel {
  flex-basis: 0 !important;
  width: 0 !important;
  max-width: 0 !important;
  padding: 0 !important;
  opacity: 0 !important;
  border-left-width: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
.module_panel_title {
  font-size: var(--dr-fs-title);
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--dr-text-faint);
  margin: 1px 2px 4px;
}
#module_panel .gr-accordion {
  margin: 0 0 4px 0 !important;
  padding: 0 !important;
  border: 1px solid var(--dr-border) !important;
  border-radius: 7px !important;
  background: var(--dr-bg-elevated) !important;
  box-shadow: none !important;
  overflow: hidden !important;
  /* overflow:hidden makes a flex item's min-height:auto resolve to 0, so once
     several modules are open the flexbox shrink algorithm was compressing —
     and overlapping — later accordions instead of letting the panel scroll. */
  flex-shrink: 0 !important;
}
#module_panel .gr-accordion > .label-wrap {
  padding: 5px 8px !important;
  font-size: var(--dr-fs-label) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  color: var(--dr-text) !important;
  margin: 0 !important;
  min-height: 0 !important;
  border-bottom: 1px solid transparent !important;
}
#module_panel .gr-accordion > .label-wrap:hover {
  background: var(--dr-bg-hover) !important;
}
#module_panel .gr-accordion .label-wrap .icon svg { stroke: var(--dr-text-dim) !important; }
#module_panel .gr-accordion[data-mod-open="1"] > .label-wrap {
  border-bottom-color: var(--dr-border) !important;
  color: var(--dr-accent-strong) !important;
}
#module_panel .gr-accordion[data-mod-open="1"] > .label-wrap .icon svg {
  stroke: var(--dr-accent-strong) !important;
}
#module_panel .gr-accordion .form {
  padding: 1px 5px 4px !important;
  gap: 1px !important;
  border: none !important;
  background: transparent !important;
}
#module_panel .block {
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
}
#module_panel label,
#module_panel label span,
#module_panel .label-wrap span,
#module_panel .head span,
#module_panel [data-testid="block-label"],
#module_panel [data-testid="block-label"] span,
#module_panel .head {
  font-size: var(--dr-fs-label) !important;
  line-height: 1.1 !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--dr-text-dim) !important;
}
/* Slider captions must not clip the way the drawer's did. */
#module_panel .head label,
#module_panel .head label span {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 100% !important;
}
#module_panel button {
  min-height: 20px !important;
  height: 20px !important;
  font-size: var(--dr-fs-control) !important;
  padding: 1px 5px !important;
  border-radius: 5px !important;
}
#module_panel .prose,
#module_panel .prose p,
#module_panel .md {
  font-size: var(--dr-fs-note) !important;
  line-height: 1.2 !important;
  margin: 1px 0 !important;
}
#module_panel input, #module_panel select {
  font-size: var(--dr-fs-control) !important;
  min-height: 18px !important;
  height: 18px !important;
  padding: 0 4px !important;
}
#module_panel input[type="range"] { height: 12px !important; }
#module_panel .head input[type="number"] { width: 40px !important; }
#module_panel .min_value, #module_panel .max_value { font-size: var(--dr-fs-tiny) !important; }
#module_panel .icon-button-wrapper,
#module_panel .reset-button { transform: scale(0.75) !important; }
/* Radio pills wrap two-up instead of one tall column per option. */
#module_panel fieldset .wrap,
#module_panel fieldset > div {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 2px !important;
}
#module_panel fieldset label {
  padding: 1px 5px !important;
  font-size: var(--dr-fs-note) !important;
  line-height: 1.15 !important;
  min-height: 0 !important;
  flex: 0 1 auto !important;
  border-radius: 4px !important;
}
#module_panel fieldset label input[type="radio"] {
  width: 9px !important;
  height: 9px !important;
  min-height: 0 !important;
  margin-right: 3px !important;
}
/* Gradio wraps an Image in a <button>; the module-panel button height rule
   above squeezed the curve plot to 20px. Let this one size to its box. */
#module_panel #curve_plot,
#module_panel #curve_plot .image-container,
#module_panel #curve_plot .image-frame,
#module_panel #curve_plot > button,
#module_panel #curve_plot .image-container > button,
#module_panel #curve_plot .wrap,
#module_panel #curve_plot [data-testid="image"] {
  height: auto !important;
  min-height: 280px !important;
  max-height: none !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: var(--dr-bg-panel) !important;
  border-radius: 5px !important;
  overflow: visible !important;
  padding: 0 !important;
  flex: 0 0 auto !important;
}
#module_panel #curve_plot img {
  width: 100% !important;
  height: auto !important;
  min-height: 260px !important;
  max-height: none !important;
  object-fit: contain !important;
  cursor: zoom-in;
  display: block !important;
}
#curve_plot [data-testid="block-label"],
#curve_plot .icon-button-wrapper { display: none !important; }
#curve_summary { margin-bottom: 4px !important; }
/* Float over the print stage only — never cover the stage filmstrip
   (#seq_strip is a fixed 46px row under the preview). */
#preview_col .block:has(#spot_readout) {
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  border: none !important;
  background: transparent !important;
  pointer-events: none !important;
  z-index: 8 !important;
}
#spot_readout {
  position: absolute !important;
  left: 10px !important;
  bottom: calc(46px + 8px) !important;
  z-index: 8 !important;
  max-width: min(420px, 70%) !important;
  padding: 6px 10px !important;
  border-radius: 6px !important;
  border: 1px solid var(--dr-border) !important;
  background: rgba(18, 18, 21, 0.82) !important;
  color: var(--dr-text) !important;
  font-size: 12px !important;
  pointer-events: none !important;
}
#spot_readout p { margin: 0 !important; }
#hist_plot,
#hist_plot > div,
#hist_plot button,
#hist_plot .image-container {
  height: auto !important;
  min-height: 120px !important;
  max-height: none !important;
  width: 100% !important;
  background: var(--dr-bg-panel) !important;
}
#hist_plot img {
  width: 100% !important;
  height: auto !important;
  object-fit: contain !important;
}
#module_panel #hist_plot button { height: auto !important; min-height: 120px !important; }
#rotate_row {
  gap: 4px !important;
  margin: 2px 0 6px 0 !important;
}
#rotate_row button {
  min-width: 0 !important;
  flex: 1 1 0 !important;
  font-size: 11px !important;
  padding: 2px 4px !important;
}
#auto_straighten_btn {
  min-width: 52px !important;
  flex: 0 0 auto !important;
}


.mod-icon {
  display: inline-flex;
  vertical-align: -2px;
  margin-right: 5px;
}
.mod-icon svg {
  width: 13px;
  height: 13px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
#ctx_menu, #dl_menu {
  position: fixed;
  z-index: 100060;
  display: none;
  min-width: 190px;
  padding: 5px;
  border-radius: var(--dr-radius);
  border: 1px solid var(--dr-border-strong);
  background: var(--dr-bg-elevated);
  box-shadow: 0 12px 32px rgba(0,0,0,0.55);
}
#ctx_menu.is-open, #dl_menu.is-open { display: block; }
#ctx_menu button, #dl_menu button {
  display: block;
  width: 100%;
  text-align: left;
  background: transparent;
  border: 0;
  color: var(--dr-text-dim);
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 0.82rem;
  cursor: pointer;
}
#ctx_menu button:hover, #dl_menu button:hover { background: var(--dr-accent-soft); color: var(--dr-text); }

/* Compact leftovers from older layout ids */
#controls_col { display: contents !important; }
#db_size_readout {
  font-size: 0.62rem !important;
  opacity: 0.9;
  color: var(--dr-text-dim) !important;
  white-space: nowrap !important;
  line-height: 1 !important;
}
#db_size_readout .prose, #db_size_readout .html-container { padding: 0 !important; margin: 0 !important; }
#db_size_readout .db-size-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--dr-border);
  background: rgba(20, 20, 23, 0.82);
  backdrop-filter: blur(3px);
  white-space: nowrap;
}
#db_size_readout .db-tool-mode {
  color: var(--dr-accent-strong);
  font-weight: 600;
}
#db_size_readout .db-size-value { color: var(--dr-text); }
#db_wave_banner { flex: 0 0 auto; }
#first_print_guide { display: none !important; }

/* Crop overlay on live print */
#live_preview.crop-armed {
  position: relative !important;
  user-select: none;
  cursor: crosshair;
}
#live_preview.crop-armed .image-container,
#live_preview.crop-armed .image-frame { position: relative !important; }
#live_preview.crop-armed img { cursor: crosshair; }
#crop_overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 5;
}
#crop_overlay .crop-shade {
  position: absolute;
  background: rgba(0, 0, 0, 0.45);
  pointer-events: none;
}
#crop_overlay .crop-box {
  position: absolute;
  border: 2px solid var(--dr-accent);
  box-shadow: 0 0 0 1px rgba(0,0,0,0.55);
  pointer-events: auto;
  cursor: move;
  box-sizing: border-box;
}
#crop_overlay .crop-box::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(to right, transparent 33.33%, rgba(224,149,79,0.35) 33.33%, rgba(224,149,79,0.35) 33.66%, transparent 33.66%, transparent 66.66%, rgba(224,149,79,0.35) 66.66%, rgba(224,149,79,0.35) 67%, transparent 67%),
    linear-gradient(to bottom, transparent 33.33%, rgba(224,149,79,0.35) 33.33%, rgba(224,149,79,0.35) 33.66%, transparent 33.66%, transparent 66.66%, rgba(224,149,79,0.35) 66.66%, rgba(224,149,79,0.35) 67%, transparent 67%);
}
#crop_overlay .crop-handle {
  position: absolute;
  width: 12px;
  height: 12px;
  background: var(--dr-accent);
  border: 1px solid #1a1a1a;
  border-radius: 2px;
  box-sizing: border-box;
  pointer-events: auto;
  z-index: 2;
}
#crop_overlay .crop-handle.nw { left: -6px; top: -6px; cursor: nwse-resize; }
#crop_overlay .crop-handle.ne { right: -6px; top: -6px; cursor: nesw-resize; }
#crop_overlay .crop-handle.sw { left: -6px; bottom: -6px; cursor: nesw-resize; }
#crop_overlay .crop-handle.se { right: -6px; bottom: -6px; cursor: nwse-resize; }
#crop_overlay .crop-handle.n { left: 50%; top: -6px; margin-left: -6px; cursor: ns-resize; }
#crop_overlay .crop-handle.s { left: 50%; bottom: -6px; margin-left: -6px; cursor: ns-resize; }
#crop_overlay .crop-handle.w { left: -6px; top: 50%; margin-top: -6px; cursor: ew-resize; }
#crop_overlay .crop-handle.e { right: -6px; top: 50%; margin-top: -6px; cursor: ew-resize; }
#live_preview.inspect-armed img { cursor: zoom-in; }
/* Native image drag-and-drop fired pointercancel and killed panning after a
   single move; touch gestures would do the same on a trackpad/tablet. */
#live_preview img {
  -webkit-user-drag: none !important;
  user-select: none !important;
  -webkit-user-select: none !important;
}
#live_preview.inspect-armed,
#live_preview.inspect-armed * {
  touch-action: none !important;
}

#db_tool_cursor {
  position: fixed;
  pointer-events: none;
  z-index: 100000;
  transform: translate(-50%, -50%);
  width: 120px;
  height: 120px;
  display: none;
}
#db_tool_cursor.db-tool-preview .db-tool-fill { opacity: 0.18; }
#db_tool_cursor.db-tool-preview .db-tool-svg { opacity: 0.95; }
#db_tool_cursor .db-tool-fill {
  position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: contain; opacity: 0.35;
  filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));
}
#db_tool_cursor .db-tool-svg {
  position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible;
  filter: drop-shadow(0 0 3px rgba(0,0,0,0.9));
}
#db_tool_cursor.db-card-resting {
  animation: db-card-breathe 1.4s ease-in-out infinite;
}
#db_stamp_asset { display: none !important; }
@keyframes db-wave-pulse {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.08); }
}
@keyframes db-wave-ring {
  0%, 100% { box-shadow: 0 0 0 6px rgba(224, 149, 79, 0.18), 0 0 22px rgba(224, 149, 79, 0.3); }
  50% { box-shadow: 0 0 0 10px rgba(224, 149, 79, 0.3), 0 0 34px rgba(224, 149, 79, 0.48); }
}
@keyframes db-card-breathe {
  0%, 100% { opacity: 0.34; transform: translate(-50%, -50%) scale(0.96); }
  50% { opacity: 0.55; transform: translate(-50%, -50%) scale(1.02); }
}
body.db-exposing #live_preview,
body.db-exposing #live_preview *,
#live_preview.db-waving,
#live_preview.db-waving *,
#live_preview.db-waving img,
#live_preview.db-tool-hover,
#live_preview.db-tool-hover *,
#live_preview.db-tool-hover img {
  cursor: none !important;
}
#live_preview.db-waving {
  outline: 3px solid var(--dr-accent) !important;
  animation: db-wave-pulse 1.1s ease-in-out infinite, db-wave-ring 1.1s ease-in-out infinite;
}
.db_clock_hidden {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
"""

# Wheel / trackpad zoom + drag pan for main and inspect viewers;
# tool silhouette follows the pointer on the live print (preview + expose).
# Gradio injects launch(js=...) as a <script> text node — it must be an IIFE
# (or bare statements), not a bare () => {} which never runs.
UI_JS = """
(() => {
  // ——— Minimal lucide-style line icons for the rail (no icon font needed) ———
  const RAIL_ICONS = {
    ingest: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    develop: '<path d="M14 2v6a2 2 0 0 0 .24.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.75-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.45 15h11.1"/><path d="M8.5 2h7"/>',
    print: '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    frame: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    log: '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    new: '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
  };
  const RAIL_LABELS = { ingest: 'Upload', develop: 'Dev', print: 'Print', frame: 'Frame', log: 'Log', new: 'New', roll: 'Roll' };
  const svgWrap = (inner) =>
    `<svg class="rail-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;
  const installRailIcons = () => {
    Object.keys(RAIL_ICONS).forEach((key) => {
      const btn = document.getElementById('rail_' + key);
      if (!btn || btn.dataset.iconReady === '1') return;
      btn.dataset.iconReady = '1';
      btn.innerHTML =
        `<span class="rail-glyph">${svgWrap(RAIL_ICONS[key])}</span>` +
        `<span class="rail-label">${RAIL_LABELS[key]}</span>`;
    });
  };
  installRailIcons();
  setInterval(installRailIcons, 1200);

  // ——— Module panel header icons (Inspect / Dodge & burn / Crop & straighten) ———
  const MODULE_ICONS = {
    mod_inspect: '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16" y2="16"/>',
    mod_curves: '<path d="M3 20V4"/><path d="M3 20h18"/><path d="M4 17c4-1 6-9 8-11s4 3 8 4"/>',
    mod_dodge_burn: '<circle cx="12" cy="12" r="4"/><path d="M12 3v2"/><path d="M12 19v2"/><path d="M5 5l1.4 1.4"/><path d="M17.6 17.6 19 19"/><path d="M3 12h2"/><path d="M19 12h2"/><path d="M5 19l1.4-1.4"/><path d="M17.6 6.4 19 5"/>',
    mod_crop: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
  };
  const installModuleIcons = () => {
    Object.keys(MODULE_ICONS).forEach((id) => {
      const acc = document.getElementById(id);
      const label = acc && acc.querySelector('.label-wrap > span:first-child');
      if (!label || label.dataset.iconReady === '1') return;
      label.dataset.iconReady = '1';
      label.insertAdjacentHTML(
        'afterbegin',
        `<span class="mod-icon">${svgWrap(MODULE_ICONS[id])}</span>`
      );
    });
  };
  installModuleIcons();
  setInterval(installModuleIcons, 1200);

  // ——— Collapse toggle for the floating stage/recipe readout ———
  const installStatusToggle = () => {
    const el = document.getElementById('ritual_status');
    if (!el || el.dataset.toggleReady === '1') return;
    el.dataset.toggleReady = '1';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'status-toggle';
    btn.title = 'Minimise';
    const paint = () => {
      const collapsed = el.classList.contains('status-collapsed');
      btn.textContent = collapsed ? '▸' : '▾';
      btn.title = collapsed ? 'Expand details' : 'Minimise';
    };
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.toggle('status-collapsed');
      try { localStorage.setItem('dr_status_collapsed', el.classList.contains('status-collapsed') ? '1' : '0'); } catch (_) {}
      paint();
    });
    if (localStorage.getItem('dr_status_collapsed') === '1') el.classList.add('status-collapsed');
    paint();
    el.appendChild(btn);
  };
  installStatusToggle();
  // Gradio replaces the markdown body on every update; re-attach if it's lost.
  setInterval(() => {
    const el = document.getElementById('ritual_status');
    if (el && !el.querySelector('.status-toggle')) {
      el.dataset.toggleReady = '';
      installStatusToggle();
    }
  }, 900);

  window.__dbPos = '';
  window.__spotPos = '';

  window.__dbGetPos = () => window.__dbPos || '';
  window.__dbToolArmed = true;
  window.__dbToolScale = 1.0;

  const clampToolScale = (s) => Math.min(2.75, Math.max(0.35, s));

  const updateSizeReadout = () => {
    const el = document.querySelector('#db_size_readout .db-size-value');
    if (!el) return;
    const pct = Math.round(clampToolScale(window.__dbToolScale || 1) * 100);
    const text = pct + '%';
    if (el.textContent !== text) el.textContent = text;
  };

  const formatPos = (nx, ny) =>
    Number(nx).toFixed(4) + ',' + Number(ny).toFixed(4) + ',' + clampToolScale(window.__dbToolScale || 1).toFixed(3);

  const writePosBox = (text) => {
    // If caller passes only x,y keep current scale; formatPos used for live writes.
    window.__dbPos = text || '';
    const root = document.querySelector('#db_pos');
    if (!root) return;
    const box = root.querySelector('textarea') || root.querySelector('input');
    if (!box) return;
    if (box.value === window.__dbPos) return;
    box.value = window.__dbPos;
    // Keep Gradio's Textbox value in sync for Timer.tick inputs.
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const readPreviewTool = () => {
    const root = document.querySelector('#preview_tool');
    if (!root) return 'print';
    const checked = root.querySelector('input[type="radio"]:checked');
    if (checked && checked.value) return checked.value;
    return 'print';
  };

  const syncPreviewToolClasses = () => {
    const tool = readPreviewTool();
    document.body.dataset.previewTool = tool;
    const live = document.querySelector('#live_preview');
    if (!live) return tool;
    live.classList.toggle('crop-armed', tool === 'frame');
    live.classList.toggle('inspect-armed', tool === 'inspect');
    const overlay = document.getElementById('crop_overlay');
    if (overlay && tool !== 'frame') overlay.style.display = 'none';
    if (tool !== 'print') hideTool();
    const modeEl = document.querySelector('#db_size_readout .db-tool-mode');
    if (modeEl) {
      const label = tool === 'frame' ? 'Frame' : (tool === 'inspect' ? 'Inspect' : 'Print');
      if (modeEl.textContent !== label) modeEl.textContent = label;
    }
    return tool;
  };

  function enhance(sel) {
    const root = document.querySelector(sel);
    if (!root || root.dataset.zoomReady === '1') return;
    const findImg = () => root.querySelector('img');
    let scale = 1;
    let panX = 0, panY = 0;
    let dragging = false, lastX = 0, lastY = 0;

    const toolActive = () =>
      root.classList.contains('db-waving') ||
      root.classList.contains('db-tool-hover') ||
      document.body.classList.contains('db-exposing');

    // Keep the zoomed frame overlapping the stage so it can't be flung away.
    const clampPan = (img) => {
      if (!img) return;
      const stage = root.getBoundingClientRect();
      const w = img.offsetWidth * scale;
      const h = img.offsetHeight * scale;
      const maxX = Math.max(0, (w - stage.width) / 2);
      const maxY = Math.max(0, (h - stage.height) / 2);
      panX = Math.min(maxX, Math.max(-maxX, panX));
      panY = Math.min(maxY, Math.max(-maxY, panY));
    };

    const apply = (img) => {
      if (!img) return;
      img.style.transformOrigin = 'center center';
      img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
      img.style.maxWidth = scale > 1.02 ? 'none' : '';
      img.style.maxHeight = scale > 1.02 ? 'none' : '';
      if (toolActive()) {
        img.style.cursor = 'none';
        root.style.cursor = 'none';
        return;
      }
      const mode = readPreviewTool();
      if (mode === 'frame') {
        img.style.cursor = 'crosshair';
        return;
      }
      if (mode === 'inspect') {
        img.style.cursor = scale > 1.02 ? (dragging ? 'grabbing' : 'grab') : 'zoom-in';
        return;
      }
      img.style.cursor = scale > 1.02 ? (dragging ? 'grabbing' : 'grab') : 'zoom-in';
    };

    root.addEventListener('wheel', (e) => {
      const img = findImg();
      if (!img) return;
      const mode = syncPreviewToolClasses();
      // Frame mode: leave scroll alone (box handles do the work).
      if (sel === '#live_preview' && mode === 'frame') return;
      // Inspect mode on live: scroll always zooms (no need for Ctrl).
      if (sel === '#live_preview' && mode === 'inspect') {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        scale = Math.min(10, Math.max(0.4, scale * factor));
        if (scale <= 1.02) { panX = 0; panY = 0; }
        apply(img);
        return;
      }
      // Print mode: scroll = resize dodge/burn tool. Ctrl/Meta+scroll = zoom.
      if (sel === '#live_preview' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
        window.__dbToolScale = clampToolScale((window.__dbToolScale || 1) * factor);
        updateSizeReadout();
        const n = (() => {
          const r = img.getBoundingClientRect();
          if (r.width < 2 || r.height < 2) return null;
          const nx = (e.clientX - r.left) / r.width;
          const ny = (e.clientY - r.top) / r.height;
          if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return [0.5, 0.5];
          return [nx, ny];
        })();
        if (n) writePosBox(formatPos(n[0], n[1]));
        // Refresh outline at pointer (or resting center).
        const flag = readFlag();
        if (flag) showToolAt(e.clientX, e.clientY, flag, false);
        return;
      }
      if (root.classList.contains('db-waving')) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      scale = Math.min(10, Math.max(0.4, scale * factor));
      if (scale <= 1.02) { panX = 0; panY = 0; }
      apply(img);
    }, { passive: false });

    // The image is natively draggable, so the browser started its own
    // drag-and-drop on pointerdown and fired pointercancel — the pan died
    // after a single move. Refuse the native drag so the stream survives.
    root.addEventListener('dragstart', (e) => e.preventDefault());

    root.addEventListener('pointerdown', (e) => {
      if (root.classList.contains('db-waving')) return;
      const mode = readPreviewTool();
      if (sel === '#live_preview' && mode === 'frame') return;
      const img = findImg();
      if (!img || scale <= 1.02) return;
      e.preventDefault();
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      root.setPointerCapture?.(e.pointerId);
      apply(img);
    });
    root.addEventListener('pointermove', (e) => {
      if (!dragging || root.classList.contains('db-waving')) return;
      const img = findImg();
      if (!img) return;
      e.preventDefault();
      panX += e.clientX - lastX;
      panY += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      clampPan(img);
      apply(img);
    });
    const endDrag = () => {
      dragging = false;
      apply(findImg());
    };
    root.addEventListener('pointerup', endDrag);
    root.addEventListener('pointercancel', endDrag);
    root.addEventListener('dblclick', () => {
      if (root.classList.contains('db-waving')) return;
      scale = 1; panX = 0; panY = 0;
      apply(findImg());
    });

    const mo = new MutationObserver(() => {
      if (!root.classList.contains('db-waving')) {
        scale = 1; panX = 0; panY = 0;
      }
      apply(findImg());
    });
    mo.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    root.dataset.zoomReady = '1';
    apply(findImg());
  }

  const hideClockChrome = () => {
    document.querySelectorAll('.db_clock_hidden').forEach((el) => {
      if (el.dataset.dbClockHidden === '1') return;
      el.dataset.dbClockHidden = '1';
      el.style.cssText =
        'position:absolute;left:-9999px;width:1px;height:1px;opacity:0;overflow:hidden;pointer-events:none;';
      el.setAttribute('aria-hidden', 'true');
    });
    document.querySelectorAll('#db_actions, #controls_col').forEach((root) => {
      root.querySelectorAll('span, p, div').forEach((node) => {
        if (node.dataset.dbDurHidden === '1') return;
        const t = (node.textContent || '').trim();
        if (/^\\d+(\\.\\d+)?s$/.test(t) && node.children.length === 0) {
          node.dataset.dbDurHidden = '1';
          node.style.display = 'none';
        }
      });
    });
  };

  const normOverImg = (img, clientX, clientY) => {
    const r = img.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return null;
    const nx = (clientX - r.left) / r.width;
    const ny = (clientY - r.top) / r.height;
    if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return null;
    return [Math.min(1, Math.max(0, nx)), Math.min(1, Math.max(0, ny))];
  };

  const writeSpotBox = (nx, ny) => {
    const text = Number(nx).toFixed(4) + ',' + Number(ny).toFixed(4);
    if (text === window.__spotPos) return;
    // Throttle Gradio traffic — ~8 Hz is enough for a readout.
    const now = performance.now();
    if (window.__spotLastWrite && now - window.__spotLastWrite < 120) return;
    window.__spotLastWrite = now;
    window.__spotPos = text;
    const root = document.querySelector('#spot_pos');
    if (!root) return;
    const box = root.querySelector('textarea') || root.querySelector('input');
    if (!box || box.value === text) return;
    box.value = text;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const forceNoCursor = (live) => {
    if (!live) return;
    if (live.style.cursor !== 'none') live.style.cursor = 'none';
    live.querySelectorAll('*').forEach((el) => {
      if (el.style.cursor !== 'none') el.style.cursor = 'none';
    });
  };

  const releaseNoCursor = (live) => {
    if (!live) return;
    if (live.style.cursor === 'none') live.style.cursor = '';
    live.querySelectorAll('*').forEach((el) => {
      if (el.style.cursor === 'none') el.style.cursor = '';
    });
  };

  // ——— Persistent module panel (darktable-style) open/close helpers ———
  const isModuleOpen = (id) => {
    const acc = document.getElementById(id);
    const lw = acc && acc.querySelector('.label-wrap');
    return !!(lw && lw.classList.contains('open'));
  };
  const openModule = (id) => {
    const acc = document.getElementById(id);
    const lw = acc && acc.querySelector('.label-wrap');
    if (!lw) return;
    if (!lw.classList.contains('open')) lw.click();
    setTimeout(() => {
      try { acc.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } catch (_) {}
    }, 60);
  };
  // Module accordion bodies mount lazily on first open, so a control inside
  // one may not exist in the DOM the instant its module opens. Poll briefly.
  const waitForEl = (selector, fn, tries = 10) => {
    const el = document.querySelector(selector);
    if (el) {
      fn(el);
      return;
    }
    if (tries > 0) setTimeout(() => waitForEl(selector, fn, tries - 1), 60);
  };

  // The dodge/burn card only takes over the pointer once that module is open
  // (right-click → Dodge / Burn, or expanding it in the panel) or a pass is running.
  const dbEngaged = (flag) => {
    if (isModuleOpen('mod_dodge_burn')) return true;
    const f = flag || readFlag();
    return !!(f && f.exposing);
  };

  const shapePaths = (kind, stroke) => {
    const fill = stroke + '44';
    const common = `fill="${fill}" stroke="${stroke}" stroke-width="3.5" stroke-linejoin="round"`;
    const k = (kind || 'soft_oval').toLowerCase();
    if (k === 'circle' || k === 'round') {
      return `<ellipse cx="50" cy="50" rx="40" ry="40" ${common} />`;
    }
    if (k === 'finger' || k === 'wand') {
      return `<ellipse cx="50" cy="50" rx="20" ry="42" ${common} />`;
    }
    if (k === 'card' || k === 'rect' || k === 'rectangle') {
      return `<rect x="14" y="20" width="72" height="60" rx="3" ${common} />`;
    }
    if (k === 'custom') {
      return `<ellipse cx="50" cy="50" rx="42" ry="34" ${common} stroke-dasharray="5 4" />`;
    }
    // soft oval default card
    return `<ellipse cx="50" cy="50" rx="44" ry="30" ${common} />`;
  };

  const ensureToolCursor = () => {
    let el = document.getElementById('db_tool_cursor');
    if (!el) {
      el = document.createElement('div');
      el.id = 'db_tool_cursor';
      el.innerHTML = '<img class="db-tool-fill" alt="" /><svg class="db-tool-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"></svg>';
      document.body.appendChild(el);
    }
    return el;
  };

  const hideTool = () => {
    const tool = document.getElementById('db_tool_cursor');
    if (tool) tool.style.display = 'none';
    const live = document.querySelector('#live_preview');
    if (live) {
      live.classList.remove('db-tool-hover');
      releaseNoCursor(live);
    }
  };

  const readCheckedValue = (allowed) => {
    for (const input of document.querySelectorAll('#controls_col input[type="radio"]:checked')) {
      const v = (input.value || '').toLowerCase();
      if (allowed.has(v)) return input.value;
    }
    return '';
  };

  const readFlag = () => {
    const flagRoot = document.querySelector('#db_flag');
    const node = flagRoot
      ? (flagRoot.getAttribute('data-exposing') != null
          ? flagRoot
          : flagRoot.querySelector('[data-exposing]'))
      : null;
    const asset = flagRoot && flagRoot.querySelector('#db_stamp_asset, img');
    const shapeFromUi = readCheckedValue(new Set(['soft_oval','circle','finger','card','custom']));
    const modeFromUi = readCheckedValue(new Set(['dodge','burn']));
    const shape = (node && node.getAttribute('data-shape')) || shapeFromUi || 'soft_oval';
    let mode = (node && node.getAttribute('data-mode')) || modeFromUi || 'burn';
    if (!mode) mode = 'burn';
    return {
      exposing: !!(node && node.getAttribute('data-exposing') === '1'),
      shape,
      mode,
      frac: parseFloat((node && node.getAttribute('data-stamp-fw')) || '0.28'),
      stamp: asset && asset.getAttribute('src') ? asset.getAttribute('src') : '',
      node,
    };
  };

  const showToolAt = (clientX, clientY, flag, resting) => {
    const live = document.querySelector('#live_preview');
    const img = live && live.querySelector('img');
    const tool = ensureToolCursor();
    if (!flag || !img) {
      hideTool();
      return;
    }
    const r = img.getBoundingClientRect();
    const frac = Math.min(0.55, Math.max(0.12, flag.frac || 0.28));
    const toolScale = clampToolScale(window.__dbToolScale || 1);
    const size = Math.max(40, frac * Math.min(r.width, r.height) * 1.15 * toolScale);
    const stroke = (flag.mode || '').toLowerCase().startsWith('dodge') ? '#6fd1c7' : '#e0954f';
    const svg = tool.querySelector('.db-tool-svg');
    const fillImg = tool.querySelector('.db-tool-fill');
    if (svg) svg.innerHTML = shapePaths(flag.shape, stroke);
    if (fillImg) {
      if (flag.stamp && flag.exposing) {
        if (fillImg.getAttribute('src') !== flag.stamp) fillImg.setAttribute('src', flag.stamp);
        fillImg.style.display = 'block';
      } else {
        fillImg.style.display = 'none';
      }
    }
    tool.style.width = size + 'px';
    tool.style.height = size + 'px';
    tool.style.left = clientX + 'px';
    tool.style.top = clientY + 'px';
    tool.classList.toggle('db-card-resting', !!resting);
    tool.classList.toggle('db-tool-preview', !flag.exposing);
    tool.style.display = 'block';
    live.classList.add('db-tool-hover');
    forceNoCursor(live);
  };

  const placeRestingCard = (live, flag) => {
    const img = live && live.querySelector('img');
    if (!img || !flag) {
      hideTool();
      return;
    }
    const r = img.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    showToolAt(r.left + r.width / 2, r.top + r.height / 2, flag, true);
    if (flag.exposing && !window.__dbPos) writePosBox(formatPos(0.5, 0.5));
  };

  const syncWave = () => {
    const flag = readFlag();
    const live = document.querySelector('#live_preview');
    const exposing = !!(flag && flag.exposing);
    const wasExposing = document.body.classList.contains('db-exposing');
    document.body.classList.toggle('db-exposing', exposing);
    if (syncPreviewToolClasses() !== 'print' || !dbEngaged(flag)) {
      if (live) live.classList.remove('db-waving');
      hideTool();
      return;
    }
    if (!exposing) {
      // Keep last nx,ny,scale so Start / scroll size survive leaving the print.
      window.__dbScrolled = false;
      if (live) live.classList.remove('db-waving');
      // Keep a faint resting preview on the print so the card is visible
      // before Start — pointer hover still moves it.
      if (live && live.querySelector('img') && !window.__dbHoveringPrint) {
        placeRestingCard(live, flag);
      }
      return;
    }
    if (live) {
      live.classList.add('db-waving');
      forceNoCursor(live);
      if (!window.__dbScrolled) {
        live.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        window.__dbScrolled = true;
      }
    }
    if (!wasExposing) {
      // Preserve scroll-wheel tool size when the timer arms.
      writePosBox(formatPos(0.5, 0.5));
    }
    if (!window.__dbHoveringPrint) placeRestingCard(live, flag);
  };

  const onLivePointer = (e) => {
    if (syncPreviewToolClasses() !== 'print') {
      window.__dbHoveringPrint = false;
      hideTool();
      return;
    }
    const flag = readFlag();
    if (!dbEngaged(flag)) {
      window.__dbHoveringPrint = false;
      hideTool();
      return;
    }
    const live = document.querySelector('#live_preview');
    const img = live && live.querySelector('img');
    if (!flag || !img) {
      window.__dbHoveringPrint = false;
      hideTool();
      return;
    }
    const n = normOverImg(img, e.clientX, e.clientY);
    if (!n) {
      window.__dbHoveringPrint = false;
      if (flag.exposing) {
        if (!window.__dbPos) placeRestingCard(live, flag);
      } else {
        placeRestingCard(live, flag);
      }
      return;
    }
    window.__dbHoveringPrint = true;
    forceNoCursor(live);
    if (flag.exposing) {
      writePosBox(formatPos(n[0], n[1]));
    } else {
      // Keep scale+pos warm so Start picks up the size you scrolled to.
      writePosBox(formatPos(n[0], n[1]));
    }
    writeSpotBox(n[0], n[1]);
    showToolAt(e.clientX, e.clientY, flag, false);
  };

  // Spot readout even when dodge/burn card is not engaged.
  const onSpotMove = (e) => {
    const live = document.querySelector('#live_preview');
    const img = live && live.querySelector('img');
    if (!img) return;
    const flag = readFlag();
    if (flag && flag.exposing) return;
    const n = normOverImg(img, e.clientX, e.clientY);
    if (!n) return;
    writeSpotBox(n[0], n[1]);
  };

  let bootScheduled = false;
  let bootLastRun = 0;
  const BOOT_MIN_GAP = 250; // ms — our own class/style writes must not re-arm us
  const boot = () => {
    // Coalesce observer storms — never re-enter synchronously.
    if (bootScheduled) return;
    bootScheduled = true;
    const wait = Math.max(0, BOOT_MIN_GAP - (performance.now() - bootLastRun));
    setTimeout(() => {
      requestAnimationFrame(() => {
        bootScheduled = false;
        bootLastRun = performance.now();
        if (window.__dbBootLock) return;
        window.__dbBootLock = true;
        try {
          enhance('#live_preview');
          enhance('#inspect_preview');
          const liveSpot = document.querySelector('#live_preview');
          if (liveSpot && liveSpot.dataset.spotReady !== '1') {
            liveSpot.dataset.spotReady = '1';
            liveSpot.addEventListener('mousemove', onSpotMove, { passive: true });
          }
          hideClockChrome();
          syncWave();
        } finally {
          window.__dbBootLock = false;
        }
      });
    }, wait);
  };
  boot();
  updateSizeReadout();
  // Only track the print for tool cursor — never hijack control clicks (upload, etc.).
  const liveRoot = () => document.querySelector('#live_preview');
  document.addEventListener('pointermove', (e) => {
    const live = liveRoot();
    if (!live || !live.contains(e.target)) {
      if (window.__dbHoveringPrint) {
        window.__dbHoveringPrint = false;
        const flag = readFlag();
        if (flag && live && live.querySelector('img')) placeRestingCard(live, flag);
      }
      return;
    }
    onLivePointer(e);
  }, { passive: true });
  document.addEventListener('pointerdown', (e) => {
    const live = liveRoot();
    if (!live || !live.contains(e.target)) return;
    onLivePointer(e);
  }, { passive: true });
  // Observe only flag + live preview. Do NOT watch #controls_col — style tweaks
  // there used to re-enter MutationObserver and freeze Upload / Commit clicks.
  const observeRoots = () => {
    // Watch 'src' only on the print: we write classes/styles there ourselves,
    // and reacting to them re-armed this observer every frame (page freeze).
    const specs = [
      ['#db_flag', { childList: true, subtree: true, attributes: true, attributeFilter: ['data-exposing', 'data-shape', 'data-mode', 'data-stamp-fw', 'src'] }],
      ['#live_preview', { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] }],
    ];
    specs.forEach(([sel, opts]) => {
      const el = document.querySelector(sel);
      if (el && el.dataset.dbObs !== '1') {
        el.dataset.dbObs = '1';
        new MutationObserver(boot).observe(el, opts);
      }
    });
  };
  observeRoots();
  setInterval(observeRoots, 2000);

  // ——— Interactive crop box on #live_preview (Frame tool mode) ———
  const CROP_MIN = 0.04; // minimum normalized side
  window.__cropBox = window.__cropBox || { x: 0, y: 0, w: 1, h: 1 };

  const readCropRatio = () => {
    const root = document.querySelector('#crop_ratio');
    if (!root) return 'free';
    const checked = root.querySelector('input[type="radio"]:checked');
    if (checked && checked.value) return checked.value;
    const sel = root.querySelector('select');
    if (sel && sel.value) return sel.value;
    return 'free';
  };

  const parseRatio = (key, imgAspect) => {
    const k = (key || 'free').toLowerCase();
    if (!k || k === 'free') return null;
    if (k === 'original') return imgAspect > 0 ? imgAspect : null;
    const m = k.match(/^(\\d+(?:\\.\\d+)?):(\\d+(?:\\.\\d+)?)$/);
    if (!m) return null;
    const a = parseFloat(m[1]), b = parseFloat(m[2]);
    if (!(a > 0 && b > 0)) return null;
    return a / b;
  };

  const writeCropRectBox = () => {
    const b = window.__cropBox;
    const text = [b.x, b.y, b.w, b.h].map((v) => Number(v).toFixed(5)).join(',');
    window.__cropRect = text;
    const root = document.querySelector('#crop_rect');
    if (!root) return;
    const box = root.querySelector('textarea') || root.querySelector('input');
    if (!box) return;
    if (box.value === text) return;
    box.value = text;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const largestBoxForRatio = (ratio) => {
    if (!ratio || !(ratio > 0)) return { x: 0, y: 0, w: 1, h: 1 };
    if (ratio >= 1) {
      const h = Math.min(1, 1 / ratio);
      return { x: 0, y: (1 - h) / 2, w: 1, h };
    }
    const w = Math.min(1, ratio);
    return { x: (1 - w) / 2, y: 0, w, h: 1 };
  };

  const clampBox = (box) => {
    let { x, y, w, h } = box;
    w = Math.max(CROP_MIN, Math.min(1, w));
    h = Math.max(CROP_MIN, Math.min(1, h));
    x = Math.max(0, Math.min(1 - w, x));
    y = Math.max(0, Math.min(1 - h, y));
    return { x, y, w, h };
  };

  const applyAspectToBox = (box, ratio, anchor) => {
    if (!ratio || !(ratio > 0)) return clampBox(box);
    let { x, y, w, h } = box;
    const fromW = () => {
      h = w / ratio;
      if (anchor === 's' || anchor === 'se' || anchor === 'sw') {
        y = box.y + box.h - h;
      } else if (!(anchor === 'n' || anchor === 'ne' || anchor === 'nw')) {
        y = box.y + (box.h - h) / 2;
      }
    };
    const fromH = () => {
      w = h * ratio;
      if (anchor === 'e' || anchor === 'ne' || anchor === 'se') {
        x = box.x + box.w - w;
      } else if (!(anchor === 'w' || anchor === 'nw' || anchor === 'sw')) {
        x = box.x + (box.w - w) / 2;
      }
    };
    if (anchor === 'n' || anchor === 's') fromH();
    else if (anchor === 'e' || anchor === 'w') fromW();
    else fromW();
    let out = clampBox({ x, y, w, h });
    const cur = out.w / Math.max(out.h, 1e-9);
    if (Math.abs(cur - ratio) > 0.01) {
      out = clampBox(largestBoxForRatio(ratio));
      const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
      out.x = Math.max(0, Math.min(1 - out.w, cx - out.w / 2));
      out.y = Math.max(0, Math.min(1 - out.h, cy - out.h / 2));
    }
    return out;
  };

  const syncOverlay = () => {
    if (window.__cropOverlaySyncing) return;
    window.__cropOverlaySyncing = true;
    try {
    const stage = document.querySelector('#live_preview');
    if (!stage || syncPreviewToolClasses() !== 'frame') {
      const overlay = document.getElementById('crop_overlay');
      if (overlay) overlay.style.display = 'none';
      return;
    }
    const img = stage.querySelector('img');
    let overlay = document.getElementById('crop_overlay');
    if (!img || img.naturalWidth < 2) {
      if (overlay) overlay.style.display = 'none';
      return;
    }
    const host = img.parentElement || stage;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'crop_overlay';
      overlay.innerHTML = `
        <div class="crop-shade" data-side="top"></div>
        <div class="crop-shade" data-side="left"></div>
        <div class="crop-shade" data-side="right"></div>
        <div class="crop-shade" data-side="bottom"></div>
        <div class="crop-box">
          <div class="crop-handle nw" data-h="nw"></div>
          <div class="crop-handle n" data-h="n"></div>
          <div class="crop-handle ne" data-h="ne"></div>
          <div class="crop-handle e" data-h="e"></div>
          <div class="crop-handle se" data-h="se"></div>
          <div class="crop-handle s" data-h="s"></div>
          <div class="crop-handle sw" data-h="sw"></div>
          <div class="crop-handle w" data-h="w"></div>
        </div>`;
      host.appendChild(overlay);
    }
    const hr = host.getBoundingClientRect();
    const ir = img.getBoundingClientRect();
    overlay.style.display = 'block';
    overlay.style.left = (ir.left - hr.left) + 'px';
    overlay.style.top = (ir.top - hr.top) + 'px';
    overlay.style.width = ir.width + 'px';
    overlay.style.height = ir.height + 'px';

    const b = window.__cropBox;
    const box = overlay.querySelector('.crop-box');
    box.style.left = (b.x * 100) + '%';
    box.style.top = (b.y * 100) + '%';
    box.style.width = (b.w * 100) + '%';
    box.style.height = (b.h * 100) + '%';

    const setShade = (side, style) => {
      const el = overlay.querySelector('.crop-shade[data-side="' + side + '"]');
      if (!el) return;
      Object.assign(el.style, style);
    };
    setShade('top', { left: '0', top: '0', width: '100%', height: (b.y * 100) + '%' });
    setShade('bottom', { left: '0', top: ((b.y + b.h) * 100) + '%', width: '100%', height: ((1 - b.y - b.h) * 100) + '%' });
    setShade('left', { left: '0', top: (b.y * 100) + '%', width: (b.x * 100) + '%', height: (b.h * 100) + '%' });
    setShade('right', { left: ((b.x + b.w) * 100) + '%', top: (b.y * 100) + '%', width: ((1 - b.x - b.w) * 100) + '%', height: (b.h * 100) + '%' });
    } finally {
      window.__cropOverlaySyncing = false;
    }
  };

  const readBoxFromInput = () => {
    const root = document.querySelector('#crop_rect');
    const box = root && (root.querySelector('textarea') || root.querySelector('input'));
    if (!box || !box.value) return;
    const parts = box.value.split(',').map(parseFloat);
    if (parts.length >= 4 && parts.every((n) => Number.isFinite(n))) {
      window.__cropBox = clampBox({ x: parts[0], y: parts[1], w: parts[2], h: parts[3] });
    }
  };

  const setupCropTool = () => {
    const stage = document.querySelector('#live_preview');
    if (!stage) return;
    syncPreviewToolClasses();
    if (stage.dataset.cropReady === '1') {
      readBoxFromInput();
      syncOverlay();
      return;
    }
    stage.dataset.cropReady = '1';
    readBoxFromInput();
    writeCropRectBox();
    syncOverlay();

    let mode = null;
    let handle = null;
    let start = null;
    let startBox = null;

    const imgAspect = () => {
      const img = stage.querySelector('img');
      if (!img || !img.clientWidth) return 1;
      return img.clientWidth / Math.max(img.clientHeight, 1);
    };

    const normFromEvent = (e) => {
      const overlay = document.getElementById('crop_overlay');
      if (!overlay || overlay.style.display === 'none') return null;
      const r = overlay.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return null;
      return [
        Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
        Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
      ];
    };

    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      if (syncPreviewToolClasses() !== 'frame') return;
      const t = e.target;
      if (!(t instanceof Element)) return;
      const h = t.getAttribute && t.getAttribute('data-h');
      const onBox = t.classList && (t.classList.contains('crop-box') || t.closest('.crop-box'));
      const n = normFromEvent(e);
      if (!n) return;
      e.preventDefault();
      stage.setPointerCapture?.(e.pointerId);
      start = n;
      startBox = { ...window.__cropBox };
      if (h) {
        mode = 'resize';
        handle = h;
      } else if (onBox && t.closest && t.closest('.crop-box') && !h) {
        if (t.classList.contains('crop-handle')) {
          mode = 'resize';
          handle = t.getAttribute('data-h');
        } else {
          mode = 'move';
          handle = null;
        }
      } else {
        mode = 'create';
        handle = 'se';
        window.__cropBox = { x: n[0], y: n[1], w: CROP_MIN, h: CROP_MIN };
        syncOverlay();
      }
    });

    stage.addEventListener('pointermove', (e) => {
      if (!mode || !start || !startBox) return;
      if (syncPreviewToolClasses() !== 'frame') return;
      const n = normFromEvent(e);
      if (!n) return;
      const dx = n[0] - start[0];
      const dy = n[1] - start[1];
      const ratio = parseRatio(readCropRatio(), imgAspect());

      if (mode === 'move') {
        window.__cropBox = clampBox({
          x: startBox.x + dx,
          y: startBox.y + dy,
          w: startBox.w,
          h: startBox.h,
        });
      } else if (mode === 'create') {
        const x0 = Math.min(start[0], n[0]);
        const y0 = Math.min(start[1], n[1]);
        const x1 = Math.max(start[0], n[0]);
        const y1 = Math.max(start[1], n[1]);
        let box = { x: x0, y: y0, w: Math.max(CROP_MIN, x1 - x0), h: Math.max(CROP_MIN, y1 - y0) };
        if (ratio) {
          const aw = Math.abs(n[0] - start[0]);
          const ah = Math.abs(n[1] - start[1]);
          let w = aw, h = ah;
          if (w / Math.max(h, 1e-9) > ratio) h = w / ratio;
          else w = h * ratio;
          const sx = n[0] >= start[0] ? 1 : -1;
          const sy = n[1] >= start[1] ? 1 : -1;
          box = {
            x: sx > 0 ? start[0] : start[0] - w,
            y: sy > 0 ? start[1] : start[1] - h,
            w, h,
          };
        }
        window.__cropBox = clampBox(box);
      } else if (mode === 'resize') {
        let box = { ...startBox };
        const H = handle;
        if (H.includes('e')) box.w = startBox.w + dx;
        if (H.includes('s')) box.h = startBox.h + dy;
        if (H.includes('w')) { box.x = startBox.x + dx; box.w = startBox.w - dx; }
        if (H.includes('n')) { box.y = startBox.y + dy; box.h = startBox.h - dy; }
        if (box.w < CROP_MIN) { if (H.includes('w')) box.x = startBox.x + startBox.w - CROP_MIN; box.w = CROP_MIN; }
        if (box.h < CROP_MIN) { if (H.includes('n')) box.y = startBox.y + startBox.h - CROP_MIN; box.h = CROP_MIN; }
        window.__cropBox = applyAspectToBox(box, ratio, H);
      }
      syncOverlay();
      writeCropRectBox();
    });

    const end = () => {
      if (!mode) return;
      mode = null; handle = null; start = null; startBox = null;
      writeCropRectBox();
      syncOverlay();
    };
    stage.addEventListener('pointerup', end);
    stage.addEventListener('pointercancel', end);

    new MutationObserver((mutations) => {
      // Ignore our own crop overlay DOM so appendChild cannot re-enter forever.
      let relevant = false;
      for (const m of mutations) {
        const t = m.target;
        if (t && (t.id === 'crop_overlay' || (t.closest && t.closest('#crop_overlay')))) continue;
        let fromOverlay = false;
        for (const n of m.addedNodes || []) {
          if (n && n.id === 'crop_overlay') { fromOverlay = true; break; }
          if (n && n.querySelector && n.querySelector('#crop_overlay')) { fromOverlay = true; break; }
        }
        if (fromOverlay) continue;
        relevant = true;
        break;
      }
      if (!relevant) return;
      readBoxFromInput();
      syncOverlay();
    }).observe(stage, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    window.addEventListener('resize', syncOverlay);
  };

  document.addEventListener('change', (e) => {
    const t = e.target;
    if (!(t instanceof Element)) return;
    if (t.closest && t.closest('#preview_tool')) {
      syncPreviewToolClasses();
      syncOverlay();
      return;
    }
    if (!t.closest || !t.closest('#crop_ratio')) return;
    const stage = document.querySelector('#live_preview');
    const img = stage && stage.querySelector('img');
    const aspect = img && img.clientWidth ? img.clientWidth / Math.max(img.clientHeight, 1) : 1;
    const ratio = parseRatio(readCropRatio(), aspect);
    if (!ratio) return;
    const next = largestBoxForRatio(ratio);
    const cx = window.__cropBox.x + window.__cropBox.w / 2;
    const cy = window.__cropBox.y + window.__cropBox.h / 2;
    next.x = Math.max(0, Math.min(1 - next.w, cx - next.w / 2));
    next.y = Math.max(0, Math.min(1 - next.h, cy - next.h / 2));
    window.__cropBox = clampBox(next);
    writeCropRectBox();
    syncOverlay();
  });

  const bootCrop = () => { try { setupCropTool(); syncPreviewToolClasses(); } catch (_) {} };
  bootCrop();
  setInterval(bootCrop, 1500);

  // ——— Fixed icon-rail workspace + right-click floating tools ———
  const setPreviewToolValue = (tool) => {
    const root = document.querySelector('#preview_tool');
    if (!root) return;
    const input = root.querySelector(`input[type="radio"][value="${tool}"]`);
    if (input && !input.checked) {
      input.checked = true;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    syncPreviewToolClasses();
  };

  const writeActiveDrawer = (name) => {
    const root = document.querySelector('#active_drawer');
    const box = root && (root.querySelector('textarea') || root.querySelector('input'));
    if (!box) return;
    if (box.value === name) return;
    box.value = name;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const readActiveDrawer = () => {
    const root = document.querySelector('#active_drawer');
    const box = root && (root.querySelector('textarea') || root.querySelector('input'));
    return (box && box.value) || document.body.dataset.drawer || 'ingest';
  };

  window.__drawerCollapsed = false;
  const applyDrawer = (name, { fromServer = false } = {}) => {
    const n = (name || 'ingest').toLowerCase();
    if (!fromServer && document.body.dataset.drawer === n && !document.body.classList.contains('drawer-collapsed')) {
      document.body.classList.add('drawer-collapsed');
      window.__drawerCollapsed = true;
      document.querySelectorAll('#icon_rail .rail-btn').forEach((b) => b.classList.remove('rail-active'));
      return;
    }
    document.body.classList.remove('drawer-collapsed');
    window.__drawerCollapsed = false;
    document.body.dataset.drawer = n;
    document.querySelectorAll('.drawer-panel').forEach((el) => {
      el.classList.toggle('is-open', el.id === 'drawer_' + n);
    });
    document.querySelectorAll('#icon_rail .rail-btn').forEach((b) => {
      const id = (b.id || '').replace('rail_', '');
      b.classList.toggle('rail-active', id === n);
    });
    if (n === 'frame') {
      setPreviewToolValue('frame');
      try { openModule('mod_crop'); } catch (_) {}
    } else if (n === 'print') {
      setPreviewToolValue('print');
    }
  };

  // Only one module is open at a time, so the panel never needs to scroll.
  const collapseOtherModules = (keepId) => {
    document.querySelectorAll('#module_panel .gr-accordion').forEach((other) => {
      if (other.id === keepId) return;
      const olw = other.querySelector('.label-wrap');
      if (olw && olw.classList.contains('open')) olw.click();
      other.setAttribute('data-mod-open', '0');
    });
  };

  // Modules react to their own open/close (arm crop overlay, show the resting
  // dodge/burn card) whether toggled by hand or by the context-menu shortcut.
  document.addEventListener('click', (e) => {
    const lw = e.target && e.target.closest && e.target.closest('#module_panel .label-wrap');
    if (!lw) return;
    const acc = lw.closest('.gr-accordion');
    if (!acc) return;
    const id = acc.id;
    setTimeout(() => {
      const open = lw.classList.contains('open');
      acc.setAttribute('data-mod-open', open ? '1' : '0');
      if (open) collapseOtherModules(id);
      if (id === 'mod_crop') {
        if (open) {
          setPreviewToolValue('frame');
          syncOverlay();
        } else if (readPreviewTool() === 'frame') {
          setPreviewToolValue('print');
          syncOverlay();
        }
      } else if (id === 'mod_dodge_burn') {
        if (open) {
          try { syncWave(); } catch (_) {}
        } else {
          hideTool();
        }
      } else if (id === 'mod_curves' && open) {
        // Nudge the hidden box so the server rebuilds the plot on open.
        const root = document.querySelector('#curves_open');
        const box = root && (root.querySelector('textarea') || root.querySelector('input'));
        if (box) {
          box.value = String(Date.now());
          box.dispatchEvent(new Event('input', { bubbles: true }));
          box.dispatchEvent(new Event('change', { bubbles: true }));
        }
      } else if (id === 'mod_inspect' && open) {
        const root = document.querySelector('#inspect_open');
        const box = root && (root.querySelector('textarea') || root.querySelector('input'));
        if (box) {
          box.value = String(Date.now());
          box.dispatchEvent(new Event('input', { bubbles: true }));
          box.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    }, 30);
  });

  const ensureCtxMenu = () => {
    let menu = document.getElementById('ctx_menu');
    if (menu) return menu;
    menu = document.createElement('div');
    menu.id = 'ctx_menu';
    menu.innerHTML = `
      <button type="button" data-act="zoom">Inspect · zoom</button>
      <button type="button" data-act="dodge">Dodge</button>
      <button type="button" data-act="burn">Burn</button>
      <button type="button" data-act="crop">Crop & straighten</button>
      <button type="button" data-act="autostraighten">Auto straighten</button>
      <button type="button" data-act="autocrop">Auto crop</button>
    `;
    document.body.appendChild(menu);
    menu.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      e.preventDefault();
      e.stopPropagation();
      const act = btn.getAttribute('data-act');
      menu.classList.remove('is-open');
      // Defer all work off the click stack — building the crop overlay (or
      // anything else) synchronously during the menu click freezes the tab.
      if (act === 'zoom') {
        setTimeout(() => {
          setPreviewToolValue('inspect');
          openModule('mod_inspect');
        }, 0);
      } else if (act === 'dodge' || act === 'burn') {
        setTimeout(() => {
          setPreviewToolValue('print');
          openModule('mod_dodge_burn');
          // Accordion bodies mount lazily on first open — the radio input
          // doesn't exist in the DOM yet the instant we open it, so poll.
          waitForEl(`#mod_dodge_burn input[type="radio"][value="${act}"]`, (input) => {
            if (!input.checked) input.click();
          });
        }, 0);
      } else if (act === 'crop') {
        setTimeout(() => openModule('mod_crop'), 0);
      } else if (act === 'autostraighten') {
        setTimeout(() => {
          openModule('mod_crop');
          waitForEl('#auto_straighten_btn:not(:disabled)', (btn) => btn.click());
        }, 0);
      } else if (act === 'autocrop') {
        // Never find buttons by label "Auto crop" — that matches this menu item
        // and recurses until the tab freezes.
        setTimeout(() => {
          openModule('mod_crop');
          waitForEl('#auto_crop_btn:not(:disabled)', (autoBtn) => autoBtn.click());
        }, 0);
      }
    });
    return menu;
  };

  // ——— Download popup: one trigger, a menu when there's more than one package ———
  const DOWNLOAD_LABELS = {
    print: 'Print only',
    both: 'Print + negative',
    negative: 'Negative only',
  };
  const clickPackage = (mode) => {
    const host = document.getElementById('dl_pkg_' + mode);
    if (!host) return;
    const btn = host.matches('button') ? host : host.querySelector('button');
    if (btn) btn.click();
  };
  const readDownloadModes = () => {
    const root = document.querySelector('#download_modes');
    const box = root && (root.querySelector('textarea') || root.querySelector('input'));
    return ((box && box.value) || '').split(',').map((m) => m.trim()).filter(Boolean);
  };
  const ensureDownloadMenu = () => {
    let menu = document.getElementById('dl_menu');
    if (menu) return menu;
    menu = document.createElement('div');
    menu.id = 'dl_menu';
    menu.className = 'ctx-menu';
    document.body.appendChild(menu);
    menu.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-mode]');
      if (!b) return;
      e.preventDefault();
      e.stopPropagation();
      menu.classList.remove('is-open');
      clickPackage(b.getAttribute('data-mode'));
    });
    return menu;
  };
  document.addEventListener('click', (e) => {
    const trigger = e.target && e.target.closest && e.target.closest('#download_trigger');
    if (!trigger) {
      const open = document.getElementById('dl_menu');
      if (open && !e.target.closest('#dl_menu')) open.classList.remove('is-open');
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    const modes = readDownloadModes();
    if (modes.length <= 1) {
      clickPackage(modes[0] || 'negative');
      return;
    }
    const menu = ensureDownloadMenu();
    menu.innerHTML = modes
      .map((m) => `<button type="button" data-mode="${m}">${DOWNLOAD_LABELS[m] || m}</button>`)
      .join('');
    menu.classList.add('is-open');
    const r = trigger.getBoundingClientRect();
    const pad = 8;
    menu.style.left = Math.min(r.left, window.innerWidth - 190) + 'px';
    // Prefer opening below the button, flip above when there's no room.
    requestAnimationFrame(() => {
      const h = menu.getBoundingClientRect().height;
      const below = r.bottom + 4;
      menu.style.top = (below + h > window.innerHeight - pad
        ? Math.max(pad, r.top - h - 4)
        : below) + 'px';
    });
  });

  document.addEventListener('contextmenu', (e) => {
    const live = document.querySelector('#live_preview');
    if (!live || !live.contains(e.target)) return;
    e.preventDefault();
    const menu = ensureCtxMenu();
    menu.dataset.x = String(e.clientX);
    menu.dataset.y = String(e.clientY);
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.add('is-open');
  });
  document.addEventListener('click', (e) => {
    const menu = document.getElementById('ctx_menu');
    if (menu && !menu.contains(e.target)) menu.classList.remove('is-open');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const menu = document.getElementById('ctx_menu');
      if (menu) menu.classList.remove('is-open');
    }
  });

  // Rail clicks (toggle collapse if same icon)
  document.addEventListener('click', (e) => {
    const btn = e.target && e.target.closest && e.target.closest('#icon_rail button.rail-btn');
    if (!btn) return;
    const name = (btn.id || '').replace('rail_', '');
    if (!name || name === 'new') return;
    applyDrawer(name, { fromServer: false });
    writeActiveDrawer(name);
  });

  // Rail clicks also toggle via Gradio → active_drawer; observe that box.
  let lastDrawerVal = '';
  const syncDrawerFromBox = () => {
    const v = readActiveDrawer();
    if (v && v !== lastDrawerVal) {
      lastDrawerVal = v;
      applyDrawer(v, { fromServer: true });
    }
  };
  setInterval(syncDrawerFromBox, 400);
  applyDrawer(readActiveDrawer() || 'ingest', { fromServer: true });

  // Camera roll HTML: real ✕ buttons (data-roll-remove) + frame clicks
  // (data-roll-switch). Write into off-screen Gradio inputs with the native
  // value setter so Svelte/Gradio actually records the change — then the
  // textbox .change handlers run (do NOT also click the button; that races
  // and often submits the previous index).
  const setGradioValue = (rootId, value) => {
    const root = document.querySelector(rootId);
    if (!root) return false;
    const box = root.querySelector('textarea, input');
    if (!box) return false;
    const proto = box.tagName === 'TEXTAREA'
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(box, String(value));
    else box.value = String(value);
    box.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertFromPaste' }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };
  const clickHiddenBtn = (rootId) => {
    const root = document.querySelector(rootId);
    const btn = root && (root.querySelector('button') || root);
    if (btn && typeof btn.click === 'function') btn.click();
  };
  if (!window.__rollClickBound) {
    window.__rollClickBound = true;
    document.addEventListener('click', (e) => {
      const t = e.target;
      if (!t || !t.closest) return;
      // Match the attributes directly — nested "#camera_roll [data-…]" in
      // closest() is unreliable across Gradio's wrapper DOM.
      const rm = t.closest('[data-roll-remove]');
      if (rm) {
        e.preventDefault();
        e.stopPropagation();
        const idx = rm.getAttribute('data-roll-remove');
        if (idx == null) return;
        if (!setGradioValue('#roll_remove_index', idx + ':' + Date.now())) {
          setTimeout(() => clickHiddenBtn('#roll_remove'), 0);
        }
        return;
      }
      const sw = t.closest('[data-roll-switch]');
      if (sw) {
        e.preventDefault();
        e.stopPropagation();
        const idx = sw.getAttribute('data-roll-switch');
        if (idx == null) return;
        // Tokenized so re-clicking the same frame still fires .change.
        if (!setGradioValue('#roll_switch_index', idx + ':' + Date.now())) {
          setTimeout(() => clickHiddenBtn('#roll_switch'), 0);
        }
      }
    }, true);
  }

  // Fit the live print stage to remaining #preview_col space.
  // Do NOT MutationObserver 'style' — writing heights would re-enter forever.
  const setStyleIfChanged = (el, prop, value) => {
    if (!el) return;
    if (el.style[prop] !== value) el.style[prop] = value;
  };
  const fitLiveStage = () => {
    if (window.__fitLiveBusy) return;
    window.__fitLiveBusy = true;
    try {
      const col = document.querySelector('#preview_col');
      const live = document.querySelector('#live_preview');
      if (!col || !live) return;
      const colRect = col.getBoundingClientRect();
      if (colRect.height < 80) return;
      // #db_size_readout is absolutely positioned over the print, so it costs
      // no vertical budget. Only the wave banner and filmstrip do.
      let used = 6;
      ['#db_wave_banner', '#seq_strip'].forEach((sel) => {
        const el = document.querySelector(sel);
        if (!el) return;
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') return;
        used += el.getBoundingClientRect().height;
      });
      const h = Math.max(200, Math.floor(colRect.height - used));
      const hPx = h + 'px';
      if (live.dataset.fitH === hPx) return;
      live.dataset.fitH = hPx;
      const block = live.closest('.block') || live;
      [block, live].forEach((el) => {
        setStyleIfChanged(el, 'flex', '1 1 0');
        setStyleIfChanged(el, 'height', hPx);
        setStyleIfChanged(el, 'minHeight', hPx);
        setStyleIfChanged(el, 'maxHeight', hPx);
        setStyleIfChanged(el, 'overflow', 'hidden');
      });
      live.querySelectorAll(
        '.wrap, .image-container, .image-frame, [data-testid="image"], .image-container > div'
      ).forEach((el) => {
        if (el.id === 'crop_overlay' || (el.closest && el.closest('#crop_overlay'))) return;
        setStyleIfChanged(el, 'height', hPx);
        setStyleIfChanged(el, 'maxHeight', hPx);
        setStyleIfChanged(el, 'minHeight', '0px');
        setStyleIfChanged(el, 'width', '100%');
        setStyleIfChanged(el, 'overflow', 'hidden');
        setStyleIfChanged(el, 'display', 'flex');
        setStyleIfChanged(el, 'alignItems', 'center');
        setStyleIfChanged(el, 'justifyContent', 'center');
        setStyleIfChanged(el, 'boxSizing', 'border-box');
      });
      const img = live.querySelector('img');
      if (img) {
        setStyleIfChanged(img, 'maxHeight', hPx);
        setStyleIfChanged(img, 'maxWidth', '100%');
        setStyleIfChanged(img, 'width', 'auto');
        setStyleIfChanged(img, 'height', 'auto');
        setStyleIfChanged(img, 'objectFit', 'contain');
      }
    } finally {
      window.__fitLiveBusy = false;
    }
  };
  let fitPending = false;
  const scheduleFit = () => {
    if (fitPending) return;
    fitPending = true;
    requestAnimationFrame(() => {
      fitPending = false;
      fitLiveStage();
    });
  };
  window.addEventListener('resize', scheduleFit);
  setInterval(scheduleFit, 1200);
  scheduleFit();
  const colEl = document.querySelector('#preview_col');
  if (colEl && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(scheduleFit).observe(colEl);
  }
  const liveEl = document.querySelector('#live_preview');
  if (liveEl) {
    new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'attributes' && m.attributeName === 'src') {
          liveEl.dataset.fitH = '';
          scheduleFit();
          return;
        }
        if (m.type === 'childList') {
          for (const n of m.addedNodes || []) {
            if (n.nodeName === 'IMG' || (n.querySelector && n.querySelector('img'))) {
              liveEl.dataset.fitH = '';
              scheduleFit();
              return;
            }
          }
        }
      }
    }).observe(liveEl, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src'],
    });
  }
})();
"""

DB_TICK_JS = """
(paper, pe, pg, pc, pos) => {
  // Gradio does not pass State into js — only transform the visible inputs.
  const p = (window.__dbGetPos && window.__dbGetPos()) || pos || '';
  return [paper, pe, pg, pc, p];
}
"""


def _profile_path(paths, profile_id: str) -> Path:
    for p in paths:
        if json.loads(p.read_text(encoding="utf-8"))["id"] == profile_id:
            return p
    raise FileNotFoundError(profile_id)


def _film_profile(film_id: str):
    return load_film_profile(_profile_path(list_film_profiles(), film_id))




def _print_maps(state):
    """Reflectance / density under the current print draft."""
    draft = (state or {}).get("print_draft")
    if draft is None:
        return None, None
    return getattr(draft, "reflectance", None), getattr(draft, "print_density", None)


def _display_live_rgb(state, live=None):
    """Live RGB with optional A/B swap and clipping overlay."""
    s = state or {}
    if s.get("ab_showing") == "A" and s.get("ab_rgb") is not None:
        return s["ab_rgb"]
    rgb = live if live is not None else s.get("live_rgb")
    if rgb is None:
        return None
    if s.get("clip_hi") or s.get("clip_lo"):
        refl, _ = _print_maps(s)
        return apply_clipping_overlay(
            rgb,
            refl,
            show_highlights=bool(s.get("clip_hi")),
            show_shadows=bool(s.get("clip_lo")),
        )
    return rgb


def read_spot(spot_pos, state):
    """Zone / density under the print pointer."""
    if not state or state.get("print_draft") is None:
        return "_Develop + print preview first, then hover the print._"
    text = str(spot_pos or "").strip()
    if not text or "," not in text:
        return "_Hover the print for Zone / density._"
    try:
        parts = [p.strip() for p in text.split(",")]
        nx, ny = float(parts[0]), float(parts[1])
    except Exception:
        return "_Hover the print for Zone / density._"
    refl, dens = _print_maps(state)
    return spot_markdown(spot_at(refl, dens, nx, ny))


def refresh_inspect_tools(clip_hi, clip_lo, state):
    """Histogram + clipping overlay for the Inspect module."""
    state = {**(state or {})}
    state["clip_hi"] = bool(clip_hi)
    state["clip_lo"] = bool(clip_lo)
    refl, _ = _print_maps(state)
    hist = render_print_histogram(refl)
    live = _display_live_rgb(state)
    if hist is None:
        hist = gr.update()
    tip = (
        "_Histogram of print reflectance with Zone ticks. "
        "Clipping paints blown paper-white (red) and crushed Dmax (blue). "
        "**Fit to paper** auto-sets the timer (and softens grade if needed)._"
    )
    return hist, tip, _viewer_frame(state, live=live), state


def auto_fit_print_tones(print_exposure, print_grade, state):
    """One-click: pull blown / crushed tones back onto the paper."""
    if not state or state.get("print_draft") is None:
        raise gr.Error("Run a print preview first (Commit Develop, then adjust Print).")
    refl, _ = _print_maps(state)
    fit = suggest_tone_fit(
        refl,
        base_seconds=float(print_exposure),
        grade=float(print_grade),
    )
    if not fit.get("ok"):
        raise gr.Error(fit.get("message") or "Could not fit tones.")

    # Leave the warning overlays on so the result is immediately readable.
    hist = _history_md(state["dn"]) if state.get("dn") is not None else ""
    summary = (
        f"{_stage_banner(state.get('stage', 'print'), _locks(state))}\n\n"
        f"{fit['message']}\n\n{hist}"
    )
    state = {
        **state,
        "clip_hi": True,
        "clip_lo": True,
        "summary_cache": summary,
    }
    tip = fit["message"]
    return (
        gr.update(value=float(fit["base_seconds"])),
        gr.update(value=float(fit["grade"])),
        gr.update(value=True),  # clip_hi
        gr.update(value=True),  # clip_lo
        tip,
        state,
    )


def pin_ab_print(state):
    """Pin the current live print as reference A."""
    if not state or state.get("live_rgb") is None:
        raise gr.Error("Nothing to pin — run a print preview first.")
    state = {
        **state,
        "ab_rgb": np.asarray(state["live_rgb"]).copy(),
        "ab_showing": "live",
    }
    return (
        state,
        "**A pinned** — keep working, then toggle A / Live.",
        gr.update(interactive=True, value="Show A"),
    )


def toggle_ab_print(state):
    """Flip the large preview between pinned A and the live print."""
    if not state or state.get("ab_rgb") is None:
        raise gr.Error("Pin A first.")
    showing = "A" if state.get("ab_showing") != "A" else "live"
    state = {**state, "ab_showing": showing}
    label = "Show Live" if showing == "A" else "Show A"
    tip = (
        "**Viewing A** — pinned reference."
        if showing == "A"
        else "**Viewing Live** — current theoretical print."
    )
    return (
        _viewer_frame(state, live=_display_live_rgb(state)),
        tip,
        gr.update(value=label),
        state,
    )


def export_recipe_file(
    chemistry_mode, film_id, developer_id, development_minutes, contrast, grain,
    exposure_index, contrast_filter, scene_exposure, halation,
    paper_id, print_grade, print_exposure, print_contrast, recipe_name,
):
    recipe = build_recipe(
        film_id=film_id,
        developer_id=developer_id,
        development_minutes=float(development_minutes),
        contrast=float(contrast),
        grain=float(grain),
        paper_id=paper_id,
        print_grade=float(print_grade),
        print_exposure=float(print_exposure),
        print_contrast=float(print_contrast),
        chemistry_mode=str(chemistry_mode or "bw"),
        name=str(recipe_name or "recipe"),
        extras={
            "exposure_index": float(exposure_index),
            "contrast_filter": str(contrast_filter),
            "scene_exposure_seconds": float(scene_exposure),
            "halation": float(halation),
        },
    )
    out = Path(tempfile.gettempdir()) / "darkroom_downloads"
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in (recipe_name or "recipe"))[:48]
    path = out / f"{stem or 'recipe'}.json"
    save_recipe(path, recipe)
    return gr.update(value=str(path))


def apply_recipe_file(recipe_file, state):
    """Load a recipe JSON onto the active frame only (not the whole roll)."""
    if recipe_file is None:
        raise gr.Error("Choose a recipe JSON first.")
    path = recipe_file if isinstance(recipe_file, (str, Path)) else getattr(recipe_file, "name", None)
    if not path:
        raise gr.Error("Choose a recipe JSON first.")
    recipe = load_recipe(path)
    tip = f"**Recipe loaded** — {recipe.get('name', 'untitled')} · this frame only."
    recipe_mode = str(recipe.get("chemistry_mode") or "bw").lower()
    if recipe_mode not in {"bw", "color"}:
        recipe_mode = "bw"
    current_mode = str(((state or {}).get("controls") or {}).get("chemistry_mode") or "bw")
    if state and state.get("dn") is not None and current_mode != recipe_mode:
        raise gr.Error(
            f"Recipe is {recipe_mode.upper()} chemistry — switch Chemistry mode before loading."
        )
    film_id = recipe["film_id"]
    profile = _film_profile(film_id)
    chem = get_chemistry(profile, recipe["developer_id"])
    minutes = float(recipe["development_minutes"])
    if chem is not None:
        tmin, tmax, _normal = time_slider_bounds(chem)
        minutes = float(np.clip(minutes, tmin, tmax))
        minutes_update = gr.update(
            minimum=tmin,
            maximum=tmax,
            value=minutes,
            step=0.25,
        )
    else:
        minutes_update = gr.update(value=minutes)
    extras = recipe.get("extensions") or {}
    contrast = float(recipe.get("contrast", 0.0))
    grain = float(recipe.get("grain", 1.0))
    exposure_index = float(extras.get("exposure_index", profile.iso))
    contrast_filter = str(extras.get("contrast_filter", "none"))
    scene_exposure = float(extras.get("scene_exposure_seconds", 0.01))
    halation = float(extras.get("halation", 0.0))
    paper_id = recipe["paper_id"]
    print_grade = float(recipe["print_grade"])
    print_exposure = float(recipe["print_exposure"])
    print_contrast = float(recipe.get("print_contrast", 0.0))
    # Persist onto the active frame so switching away / back keeps the recipe
    # here without leaking it onto other roll frames.
    if state and state.get("dn") is not None:
        base = _merged_frame_controls(state)
        state = _mark_dirty(
            state,
            {
                **base,
                "chemistry_mode": recipe_mode,
                "film_id": film_id,
                "developer_id": recipe["developer_id"],
                "development_minutes": minutes,
                "contrast": contrast,
                "grain": grain,
                "exposure_index": exposure_index,
                "contrast_filter": contrast_filter,
                "scene_exposure": scene_exposure,
                "halation": halation,
                "paper_id": paper_id,
                "print_grade": print_grade,
                "print_exposure": print_exposure,
                "print_contrast": print_contrast,
            },
        )
    film_choices = FILM_CHOICES_COLOR if recipe_mode == "color" else FILM_CHOICES_BW
    paper_choices = PAPER_CHOICES_COLOR if recipe_mode == "color" else PAPER_CHOICES_BW
    return (
        gr.update(value=recipe_mode),
        gr.update(choices=film_choices, value=film_id),
        gr.update(choices=chemistry_choices(profile), value=recipe["developer_id"]),
        minutes_update,
        gr.update(value=contrast),
        gr.update(value=grain),
        gr.update(value=exposure_index),
        gr.update(value=contrast_filter),
        gr.update(value=scene_exposure),
        gr.update(value=halation),
        gr.update(choices=paper_choices, value=paper_id),
        gr.update(value=print_grade),
        gr.update(value=print_exposure),
        gr.update(value=print_contrast),
        gr.update(value=str(recipe.get("name", ""))),
        tip,
        state,
    )


def refresh_curves(
    film_id, developer_id, development_minutes, contrast, paper_id, print_grade,
    print_exposure, state,
):
    """Sample the curves currently in play and report where this frame lands."""
    if not state or state.get("dn") is None:
        return (
            gr.update(),
            "_Commit Ingest first — the scene is what makes these curves useful._",
        )
    profile = _film_profile(film_id)
    chem = get_chemistry(profile, developer_id)
    minutes = float(development_minutes) if chem is not None else None
    rel, minutes_resolved, _style = resolve_relative_time(
        profile,
        str(developer_id),
        development_minutes=minutes,
        relative_time=None if minutes is not None else 1.0,
    )
    paper = None
    if _locked(state, "development"):
        paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))

    report = build_curve_report(
        state["dn"],
        profile,
        relative_time=rel,
        contrast_modifier=float(contrast),
        developer_id=str(developer_id),
        development_minutes=minutes_resolved,
        paper=paper,
        grade=float(print_grade),
        base_exposure_seconds=float(print_exposure),
    )
    return render_curve_plot(report), curve_summary_markdown(report)


# Stable Gradio slider span covering B&W tank times (~3–22 min) and C-41/E-6
# process times (~2.5–11 min). Chem-specific normals still drive the label and
# reset value; narrowing min/max per chemistry races with cascading .change/.then
# events and raises "Value X is greater than maximum value Y" (e.g. Tri-X 7.75
# still in the payload when Color C-41 max becomes 5.5).
_DEV_TIME_SLIDER_MIN = 1.5
_DEV_TIME_SLIDER_MAX = 24.0


def _chem_time_update(film_id: str, developer_id: str, *, reset_to_normal: bool = True):
    """Gradio updates for developer dropdown / minutes slider."""
    profile = _film_profile(film_id)
    chem = get_chemistry(profile, developer_id)
    if chem is None:
        label = "Dev time (rel. ×8 min stand-in)"
        return gr.update(
            minimum=_DEV_TIME_SLIDER_MIN,
            maximum=_DEV_TIME_SLIDER_MAX,
            value=8.0,
            label=label,
            step=0.25,
        )
    _tmin, _tmax, normal = time_slider_bounds(chem)
    family = chem.get("curve_family") or []
    if isinstance(family, list) and len(family) >= 2:
        times = ", ".join(f"{float(m['minutes']):g}" for m in sorted(family, key=lambda x: x["minutes"]))
        label = f"Dev time · N={normal:g} [{times}]"
    else:
        label = f"Dev time · N={normal:g} @20°C"
    value = float(np.clip(float(normal), _DEV_TIME_SLIDER_MIN, _DEV_TIME_SLIDER_MAX))
    return gr.update(
        minimum=_DEV_TIME_SLIDER_MIN,
        maximum=_DEV_TIME_SLIDER_MAX,
        value=value,
        label=label,
        step=0.25,
    )


def on_film_change(film_id: str):
    profile = _film_profile(film_id)
    chem_id = default_chemistry_id(profile)
    choices = chemistry_choices(profile)
    return (
        gr.update(choices=choices, value=chem_id),
        _chem_time_update(film_id, chem_id, reset_to_normal=True),
        gr.update(value=float(profile.iso)),
    )


def on_developer_change(film_id: str, developer_id: str):
    return _chem_time_update(film_id, developer_id, reset_to_normal=True)


def on_chemistry_mode_change(mode: str):
    """Swap film/paper catalogs when toggling B&W vs Color Chemistry."""
    mode = str(mode or "bw").lower()
    if mode not in {"bw", "color"}:
        mode = "bw"
    film_choices = FILM_CHOICES_COLOR if mode == "color" else FILM_CHOICES_BW
    paper_choices = PAPER_CHOICES_COLOR if mode == "color" else PAPER_CHOICES_BW
    if not film_choices:
        raise gr.Error("No film profiles for that chemistry mode.")
    film_id = film_choices[0][1]
    if mode == "color":
        # Prefer a C-41 negative stock as the default color entry point.
        for _label, fid in film_choices:
            try:
                if str(_film_profile(fid).type).lower() == "color_negative":
                    film_id = fid
                    break
            except Exception:
                continue
    profile = _film_profile(film_id)
    chem_id = default_chemistry_id(profile)
    chem_choices = chemistry_choices(profile)
    paper_id = paper_choices[0][1] if paper_choices else None
    return (
        gr.update(choices=film_choices, value=film_id),
        gr.update(choices=chem_choices, value=chem_id),
        _chem_time_update(film_id, chem_id, reset_to_normal=True),
        gr.update(value=float(profile.iso)),
        gr.update(choices=paper_choices, value=paper_id),
    )


# Initial Develop controls (film-specific chemistry + datasheet normal minutes).
if FILM_CHOICES:
    _boot_film = FILM_CHOICES[0][1]
    _boot_profile = _film_profile(_boot_film)
    _INIT_DEV_CHOICES = chemistry_choices(_boot_profile)
    _INIT_DEV_ID = default_chemistry_id(_boot_profile)
    _boot_chem = get_chemistry(_boot_profile, _INIT_DEV_ID) or {
        "normal_minutes": 8.0,
        "time_min": 4.0,
        "time_max": 16.0,
    }
    _INIT_TMIN, _INIT_TMAX, _INIT_TNORM = time_slider_bounds(_boot_chem)


def _resolve_input(file_obj, sample_path: str | None) -> str | None:
    """Prefer an uploaded file over the sample dropdown when both are set."""
    paths = _collect_input_paths(file_obj, sample_path)
    return paths[0] if paths else None


def _collect_input_paths(file_obj, sample_path: str | None) -> list[str]:
    """Collect one or more upload paths; fall back to the sample when empty."""
    paths: list[str] = []
    if file_obj is not None:
        items = file_obj if isinstance(file_obj, (list, tuple)) else [file_obj]
        for item in items:
            if item is None:
                continue
            if isinstance(item, str):
                path = item
            else:
                path = getattr(item, "name", None) or str(item)
            if path:
                paths.append(path)
    if not paths and sample_path:
        paths.append(str(sample_path))
    return paths


# Per-frame working state lives on the session dict; the camera roll keeps a
# snapshot list so frames can be added, switched, and removed independently.
_ROLL_KEYS = ("roll", "roll_index")


def _frame_payload(state: dict) -> dict:
    return {k: v for k, v in state.items() if k not in _ROLL_KEYS}


def _ensure_roll(state) -> dict:
    if not state:
        return {"roll": [], "roll_index": -1}
    if "roll" in state:
        return state
    # Migrate single-frame sessions created before the camera roll existed.
    if state.get("dn") is not None:
        frame = _frame_payload(state)
        return {**state, "roll": [frame], "roll_index": 0}
    return {**state, "roll": [], "roll_index": -1}


def _sync_active_into_roll(state) -> dict:
    state = _ensure_roll(state)
    roll = list(state.get("roll") or [])
    idx = int(state.get("roll_index", -1))
    if state.get("dn") is not None and 0 <= idx < len(roll):
        payload = _frame_payload(state)
        payload["dirty"] = False
        roll[idx] = payload
        return {**state, "roll": roll, "dirty": False}
    return state


def _activate_roll_index(state, index: int, *, save_current: bool = True) -> dict:
    """Make roll[index] the working frame.

    save_current=True writes the active frame into the roll first (Save).
    save_current=False leaves the roll slot untouched (Discard).
    """
    if save_current:
        state = _sync_active_into_roll(state)
    else:
        state = _ensure_roll(state)
    roll = list(state.get("roll") or [])
    if not roll:
        return {"roll": [], "roll_index": -1, "dirty": False}
    index = max(0, min(int(index), len(roll) - 1))
    activated = dict(roll[index])
    activated["roll"] = roll
    activated["roll_index"] = index
    activated["dirty"] = False
    return activated


def _capture_controls(
    chemistry_mode,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    cc_cyan,
    cc_magenta,
    cc_yellow,
) -> dict:
    """Snapshot Develop/Print UI controls so each roll frame keeps its recipe."""
    mode = str(chemistry_mode or "bw").lower()
    if mode not in {"bw", "color"}:
        mode = "bw"
    return {
        "chemistry_mode": mode,
        "film_id": film_id,
        "developer_id": developer_id,
        "development_minutes": float(development_minutes),
        "contrast": float(contrast),
        "grain": float(grain),
        "exposure_index": float(exposure_index),
        "contrast_filter": str(contrast_filter),
        "scene_exposure": float(scene_exposure),
        "halation": float(halation),
        "paper_id": paper_id,
        "print_exposure": float(print_exposure),
        "print_grade": float(print_grade),
        "print_contrast": float(print_contrast),
        "split_grade": bool(split_on),
        "soft_grade": float(soft_grade),
        "hard_grade": float(hard_grade),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "test_strips": bool(test_strips_on),
        "test_bands": int(test_bands),
        "test_stops": float(test_stops),
        "flash_stops": float(flash_stops),
        "dry_down": float(dry_down),
        "tone": tone,
        "border_frac": float(border_frac),
        "cc_cyan": float(cc_cyan),
        "cc_magenta": float(cc_magenta),
        "cc_yellow": float(cc_yellow),
    }


_CONTROL_COUNT = 29

# Develop recipe controls — Commit Develop sets these interactive=False.
_DEV_CONTROL_KEYS = (
    "chemistry_mode",
    "film_id",
    "developer_id",
    "development_minutes",
    "contrast",
    "grain",
    "exposure_index",
    "contrast_filter",
    "scene_exposure",
    "halation",
)
# Print controls — Commit Print sets the core ones interactive=False.
_PRINT_CONTROL_KEYS = (
    "paper_id",
    "print_exposure",
    "print_grade",
    "print_contrast",
    "split_grade",
    "soft_grade",
    "hard_grade",
    "soft_seconds",
    "hard_seconds",
    "test_strips",
    "test_bands",
    "test_stops",
    "flash_stops",
    "dry_down",
    "tone",
    "border_frac",
    "cc_cyan",
    "cc_magenta",
    "cc_yellow",
)


def _default_controls_dict() -> dict:
    """Fresh Develop/Print defaults for a newly ingested roll frame."""
    return {
        "chemistry_mode": "bw",
        "film_id": FILM_CHOICES_BW[0][1] if FILM_CHOICES_BW else None,
        "developer_id": _INIT_DEV_ID,
        "development_minutes": float(_INIT_TNORM),
        "contrast": 0.0,
        "grain": 1.0,
        "exposure_index": 400.0,
        "contrast_filter": "none",
        "scene_exposure": 0.01,
        "halation": 0.0,
        "paper_id": PAPER_CHOICES_BW[0][1] if PAPER_CHOICES_BW else None,
        "print_exposure": 8.0,
        "print_grade": 2.5,
        "print_contrast": 0.0,
        "split_grade": False,
        "soft_grade": 0.0,
        "hard_grade": 5.0,
        "soft_seconds": 4.5,
        "hard_seconds": 3.5,
        "test_strips": False,
        "test_bands": 5,
        "test_stops": 0.5,
        "flash_stops": 0.0,
        "dry_down": 0.0,
        "tone": "none",
        "border_frac": 0.0,
        "cc_cyan": 0.0,
        "cc_magenta": 0.0,
        "cc_yellow": 0.0,
    }


def _merged_frame_controls(state) -> dict:
    """Per-frame controls with defaults filled in (never borrow another frame's UI)."""
    return {**_default_controls_dict(), **((state or {}).get("controls") or {})}


def _control_interactivity_updates(state):
    """Update interactive flags only — leave current widget values alone."""
    has_dn = bool(state and state.get("dn") is not None)
    dev_on = (not _locked(state, "development")) if has_dn else True
    print_on = (not _locked(state, "print")) if has_dn else True
    return tuple(gr.update(interactive=dev_on) for _ in _DEV_CONTROL_KEYS) + tuple(
        gr.update(interactive=print_on) for _ in _PRINT_CONTROL_KEYS
    )


def _control_updates(state):
    """Restore this frame's Develop/Print values + lock-matched interactivity.

    Always writes explicit values (saved controls or defaults). Leaving values
    unset used to keep the previous frame's sliders, so edits leaked across
    the camera roll.
    """
    c = _merged_frame_controls(state)
    has_dn = bool(state and state.get("dn") is not None)
    dev_on = (not _locked(state, "development")) if has_dn else True
    print_on = (not _locked(state, "print")) if has_dn else True

    mode = str(c.get("chemistry_mode") or "bw").lower()
    if mode not in {"bw", "color"}:
        mode = "bw"
    film_choices = FILM_CHOICES_COLOR if mode == "color" else FILM_CHOICES_BW
    paper_choices = PAPER_CHOICES_COLOR if mode == "color" else PAPER_CHOICES_BW
    film_id = c["film_id"]
    film_ids = {x[1] for x in film_choices}
    if film_id not in film_ids and film_choices:
        film_id = film_choices[0][1]
    developer_id = c["developer_id"]
    minutes = float(c["development_minutes"])
    try:
        profile = _film_profile(str(film_id))
        choices = chemistry_choices(profile)
        chem = get_chemistry(profile, str(developer_id))
        if chem is not None:
            _tmin, _tmax, normal = time_slider_bounds(chem)
            # Prefer saved minutes when in range of this chem; otherwise datasheet normal.
            if _tmin <= minutes <= _tmax:
                minutes_val = minutes
            else:
                minutes_val = float(normal)
            minutes_u = gr.update(
                minimum=_DEV_TIME_SLIDER_MIN,
                maximum=_DEV_TIME_SLIDER_MAX,
                value=float(np.clip(minutes_val, _DEV_TIME_SLIDER_MIN, _DEV_TIME_SLIDER_MAX)),
                step=0.25,
                interactive=dev_on,
            )
        else:
            minutes_u = gr.update(
                minimum=_DEV_TIME_SLIDER_MIN,
                maximum=_DEV_TIME_SLIDER_MAX,
                value=float(np.clip(minutes, _DEV_TIME_SLIDER_MIN, _DEV_TIME_SLIDER_MAX)),
                interactive=dev_on,
            )
        # Prefer profile default developer when switching chemistry families.
        chem_ids = {cid for _label, cid in choices}
        if developer_id not in chem_ids and choices:
            developer_id = choices[0][1]
        head = (
            gr.update(value=mode, interactive=dev_on),
            gr.update(choices=film_choices, value=film_id, interactive=dev_on),
            gr.update(choices=choices, value=developer_id, interactive=dev_on),
            minutes_u,
        )
    except Exception:
        head = (
            gr.update(value=mode, interactive=dev_on),
            gr.update(choices=film_choices, value=film_id, interactive=dev_on),
            gr.update(value=developer_id, interactive=dev_on),
            gr.update(value=minutes, interactive=dev_on),
        )

    rest_dev = tuple(
        gr.update(value=c[k], interactive=dev_on) for k in _DEV_CONTROL_KEYS[4:]
    )
    print_u = []
    for key in _PRINT_CONTROL_KEYS:
        if key == "paper_id":
            pid = c.get("paper_id")
            pids = {x[1] for x in paper_choices}
            if pid not in pids and paper_choices:
                pid = paper_choices[0][1]
            print_u.append(
                gr.update(choices=paper_choices, value=pid, interactive=print_on)
            )
        else:
            print_u.append(gr.update(value=c[key], interactive=print_on))
    return head + rest_dev + tuple(print_u)


def _session_with_controls(state, *, drawer: str | None = "roll"):
    """Roll/ingest session outputs plus Develop/Print control restore."""
    return (*_roll_session_outputs(state, drawer=drawer), *_control_updates(state))


def _attach_controls(state, *control_args):
    """Write current UI Develop/Print values onto the active frame state."""
    if not state or len(control_args) < _CONTROL_COUNT:
        return state
    return {
        **state,
        "controls": _capture_controls(*control_args[:_CONTROL_COUNT]),
    }


def _is_dirty(state) -> bool:
    return bool(state and state.get("dirty"))


def _mark_dirty(state, controls: dict | None = None):
    if not state:
        return state
    updates = {"dirty": True}
    if controls is not None:
        updates["controls"] = controls
    return {**state, **updates}


def _roll_thumb_data_url(rgb) -> str:
    """Small JPEG data-URL for the HTML camera-roll list."""
    if rgb is None:
        return ""
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    from PIL import Image

    im = Image.fromarray(arr[..., :3])
    im.thumbnail((160, 96))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _roll_gallery_update(state):
    """Server-rendered roll HTML — each frame has a real ✕ button."""
    roll = (state or {}).get("roll") or []
    if not roll:
        return '<div class="roll-empty">No frames yet.</div>'
    try:
        active = int((state or {}).get("roll_index", 0))
    except (TypeError, ValueError):
        active = 0
    parts = ['<div class="roll-list">']
    for i, frame in enumerate(roll):
        thumb = frame.get("original_view")
        if thumb is None:
            thumb = frame.get("latent_view")
        if thumb is None:
            thumb = frame.get("live_rgb")
        name = f"Frame {i + 1}"
        dn = frame.get("dn")
        if dn is not None:
            raw = dn.metadata.get("source", {}).get("original_filename") or name
            name = Path(str(raw)).name
        safe = html_lib.escape(name)
        sel = " is-active" if i == active else ""
        src = _roll_thumb_data_url(thumb)
        parts.append(
            f'<div class="roll-item{sel}" data-roll-switch="{i}" title="{safe}">'
            f'<img src="{src}" alt="{safe}" draggable="false" />'
            f'<span class="roll-cap">{i + 1}. {safe}</span>'
            f'<button type="button" class="roll-x" data-roll-remove="{i}" '
            f'title="Remove" aria-label="Remove frame {i + 1}">×</button>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _roll_meta_md(state) -> str:
    roll = (state or {}).get("roll") or []
    if not roll:
        return "_No frames yet — add photos from **Upload**._"
    idx = int(state.get("roll_index", 0)) + 1
    dn = (state or {}).get("dn")
    name = ""
    if dn is not None:
        name = dn.metadata.get("source", {}).get("original_filename") or ""
        name = Path(str(name)).name
    label = f"**{idx}/{len(roll)}**"
    if name:
        label += f" · `{name}`"
    dirty = " · unsaved" if _is_dirty(state) else ""
    return f"{label}{dirty}  \n_Tap a frame to switch · hover ✕ to remove._"


def _drawer_for_frame(state) -> str:
    locks = set(_locks(state))
    stage = (state or {}).get("stage") or "ingest"
    if "print" in locks or stage == "print":
        return "print"
    if "ingest" in locks or stage in ("development", "develop"):
        return "develop"
    return "ingest"


def _build_ingest_frame(path: str | None) -> dict:
    """Ingest one path into a fresh per-frame state payload (no roll meta)."""
    dn = ingest_path(path or None)
    ui = dn.metadata.setdefault("ui_state", {})
    stages = ui.setdefault("committed_stages", [])
    locks = ui.setdefault("locked_stages", [])
    if "ingest" not in stages:
        stages.append("ingest")
    if "ingest" not in locks:
        locks.append("ingest")
    ui["current_stage"] = "ingest"
    dn.metadata.setdefault("history", []).append(
        {
            "op": "ingest",
            "source": dn.metadata["source"]["original_filename"],
            "working_space": dn.metadata.get("ingest", {}).get("working_space"),
        }
    )

    latent_full = _to_rgb_u8(dn.to_luminance(), assume_linear=True)
    latent_view = _downscale_rgb(latent_full, LIVE_MAX_SIDE)
    latent_inspect = _downscale_rgb(latent_full, INSPECT_MAX_SIDE)
    latent_ref = _downscale_rgb(latent_full, REF_MAX_SIDE)
    original_full = original_photo_preview(path, dn_image=dn.image)
    original_view = _downscale_rgb(original_full, LIVE_MAX_SIDE)
    original_inspect = _downscale_rgb(original_full, INSPECT_MAX_SIDE)
    original_ref = _downscale_rgb(original_full, REF_MAX_SIDE)
    summary = (
        f"{_stage_banner('development', ['ingest'])}\n\n"
        f"**Upload locked** — `{dn.metadata['source']['original_filename']}`  \n"
        f"_Use the live preview tools: **Frame** to crop, **Inspect** to zoom._\n\n"
        f"{_history_md(dn)}"
    )
    return {
        "dn": dn,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "original_ref": original_ref,
        "original_view": original_view,
        "original_inspect": original_inspect,
        "geometry_base": dn.image.copy(),
        "original_base": original_full.copy(),
        "latent_ref": latent_ref,
        "latent_view": latent_view,
        "latent_inspect": latent_inspect,
        "neg_ref": None,
        "neg_view": None,
        "neg_inspect": None,
        "live_rgb": latent_view,
        "live_inspect": latent_inspect,
        "viewer_mode": "live",
        "strip_slots": list(STRIP_DEFAULT_SLOTS),
        "development": None,
        "development_full": None,
        "stage": "development",
        "summary_cache": summary,
        "source_path": path,
        "controls": _default_controls_dict(),
        "dirty": False,
    }


def _to_rgb_u8(gray_float: np.ndarray, *, assume_linear: bool = False) -> np.ndarray:
    view = linear_to_srgb(gray_float) if assume_linear else gray_float
    arr = np.asarray(view, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # Toned / colour print path — keep channels.
        return (np.clip(arr[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    g = to_u8_gray(arr)
    return np.stack([g, g, g], axis=-1)


def _downscale_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return np.ascontiguousarray(rgb[::step, ::step])


def parse_crop_rect(text) -> tuple[float, float, float, float]:
    """Parse 'x,y,w,h' normalized crop box → (left, top, right, bottom) trim fractions."""
    raw = str(text or "").strip() or DEFAULT_CROP_RECT
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 4:
        raise ValueError("crop rect needs x,y,w,h")
    x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 0.02, 1.0))
    h = float(np.clip(h, 0.02, 1.0))
    if x + w > 1.0:
        w = 1.0 - x
    if y + h > 1.0:
        h = 1.0 - y
    left = x
    top = y
    right = max(0.0, 1.0 - (x + w))
    bottom = max(0.0, 1.0 - (y + h))
    return left, top, right, bottom


def _framing_stage_preview(state, straighten_deg: float = 0.0):
    """RGB preview for the interactive crop stage (prefer original photo base)."""
    if not state:
        return None
    deg = float(straighten_deg or 0.0)
    orig = state.get("original_base")
    if orig is not None:
        src = np.asarray(orig)
        if abs(deg) >= 1e-6:
            src = straighten_image(src.astype(np.float32), deg, fill=0.0)
            src = np.clip(src, 0, 255).astype(np.uint8)
        elif src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)
        if src.ndim == 2:
            src = np.stack([src, src, src], axis=-1)
        return _downscale_rgb(src, CROP_STAGE_MAX_SIDE)
    base = state.get("geometry_base")
    if base is None and state.get("dn") is not None:
        base = state["dn"].image
    if base is None:
        return None
    img = np.asarray(base)
    if abs(deg) >= 1e-6:
        img = straighten_image(img, deg, fill=0.0)
    return _downscale_rgb(_to_rgb_u8(img, assume_linear=True), CROP_STAGE_MAX_SIDE)


def _crop_control_echo(straighten: float = 0.0, crop_rect: str = DEFAULT_CROP_RECT, ratio: str = "free"):
    return float(straighten), str(crop_rect), str(ratio or "free")


def _proxy_dn(dn: DigitalNegative, max_side: int = LIVE_MAX_SIDE) -> DigitalNegative:
    img = dn.image
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    if step == 1:
        return dn
    return DigitalNegative(
        image=np.ascontiguousarray(img[::step, ::step]),
        metadata=copy.deepcopy(dn.metadata),
    )


def _locked(state, stage: str) -> bool:
    if not state or state.get("dn") is None:
        return False
    return stage in state["dn"].metadata.get("ui_state", {}).get("locked_stages", [])


def _short_name(text: str | None, *, keep: int = 36) -> str:
    """Readable truncation for long upload filenames in the narrow sidebar."""
    s = str(text or "").strip() or "?"
    if len(s) <= keep:
        return s
    head = max(10, keep // 2 - 1)
    tail = max(10, keep - head - 1)
    return f"{s[:head]}…{s[-tail:]}"


def _history_md(dn) -> str:
    hist = dn.metadata.get("history", [])
    # Marker split by _split_summary; accordion already titled "Decision log".
    lines = [
        "---DECISION-LOG---",
        "_Locked decisions only — exploring does not write here._",
        "",
    ]
    if not hist:
        lines.append("_No locked decisions yet. Commit a stage to record it._")
    for i, h in enumerate(hist, 1):
        op = h.get("op", "?")
        if op == "ingest":
            lines.append(f"{i}. **Upload** — `{_short_name(h.get('source'), keep=42)}`")
        elif op == "develop":
            chem = h.get("developer_name") or h.get("developer_id")
            if h.get("development_minutes") is not None:
                time_bit = f"{float(h['development_minutes']):g} min"
            else:
                time_bit = f"rel={h.get('relative_time')}"
            film = _short_name(h.get("film_profile_id"), keep=28)
            lines.append(
                f"{i}. **Develop** — `{film}`  \n"
                f"   {chem} · {time_bit} · N±={h.get('contrast_modifier')} · "
                f"grain={h.get('grain_strength')}"
            )
        elif op == "print":
            db = h.get("dodge_burn") or []
            db_bit = f" · {len(db)} local pass(es)" if db else ""
            if h.get("base_exposure_seconds") is not None:
                stops = float(h.get("overall_exposure", 0.0))
                exp_bit = f"{float(h['base_exposure_seconds']):g}s (≈ {stops:+.2f} stops)"
            else:
                exp_bit = f"{h.get('overall_exposure'):+g} stops"
            paper = _short_name(h.get("paper_id"), keep=28)
            lines.append(
                f"{i}. **Print** — `{paper}`  \n"
                f"   grade {h.get('grade')} · exp {exp_bit}"
                + (f" · nudge={h.get('contrast')}" if h.get("contrast") not in (None, 0, 0.0) else "")
                + db_bit
            )
        elif op == "unlock":
            stage = h.get("stage", "?")
            label = {"development": "Develop", "print": "Print", "ingest": "Upload"}.get(stage, stage)
            lines.append(f"{i}. **← Unlocked {label}** — previous lock opened for revision")
        elif op == "rotate":
            lines.append(
                f"{i}. **Rotate** — {h.get('degrees_cw'):+g}° "
                f"(total {h.get('total_degrees')}° CW)"
            )
        elif op == "frame":
            ratio = h.get("ratio") or "free"
            lines.append(
                f"{i}. **Crop & straighten** — "
                f"straighten {float(h.get('straighten_degrees', 0)):+.2f}° · "
                f"ratio `{ratio}` · "
                f"trim L{float(h.get('crop_left', 0))*100:.0f}% "
                f"T{float(h.get('crop_top', 0))*100:.0f}% "
                f"R{float(h.get('crop_right', 0))*100:.0f}% "
                f"B{float(h.get('crop_bottom', 0))*100:.0f}%"
            )
        elif op == "frame_reset":
            lines.append(f"{i}. **Reset framing** — back to post-ingest / last 90° orientation")
        else:
            lines.append(f"{i}. **{op}**")
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lock_labels = []
    for s in ("ingest", "development", "print"):
        if s in locks:
            lock_labels.append({"ingest": "Upload", "development": "Develop", "print": "Print"}[s])
    lines.append("")
    lines.append(f"**Currently locked:** {', '.join(lock_labels) or '—'}")
    lines.append(
        f"**Process seed:** `{dn.metadata.get('process_seed')}`  \n"
        f"_Mild tank variation; same seed = repeatable._"
    )
    return "\n".join(lines)


def _stage_banner(stage: str, locked: list | None = None) -> str:
    """Ritual progress: which stage you're working, which are locked."""
    steps = [("ingest", "Upload"), ("development", "Develop"), ("print", "Print")]
    order = {"ingest": 0, "development": 1, "print": 2}
    cur = order.get(stage, -1)
    locked_set = set(locked or [])
    parts = []
    for i, (key, label) in enumerate(steps):
        n = i + 1
        done = key in locked_set
        if i == cur and not done:
            parts.append(f"**{n}. {label} — working**")
        elif i == cur and done:
            parts.append(f"**{n}. {label} — locked**")
        elif done:
            parts.append(f"{n}. {label} — locked")
        else:
            parts.append(f"{n}. {label}")
    return " → ".join(parts)


def _locks(state) -> list:
    if not state or state.get("dn") is None:
        return []
    return list(state["dn"].metadata.get("ui_state", {}).get("locked_stages", []))


def _split_summary(full: str) -> tuple[str, str]:
    """Split status blurb from decision log for separate UI panels."""
    for marker in ("---DECISION-LOG---", "### Decision log", "### Decision history"):
        if marker in full:
            status, hist = full.split(marker, 1)
            return status.strip(), hist.strip()
    return full, ""


def _color_or_bw_negative_view(development) -> np.ndarray:
    """u8 RGB for the Developed-negative strip: C-41 orange lightbox, not the scan invert.

    ``positive_preview`` for C-41 is an inverted inspection positive — putting that
    in the negative slot made Live print and Negative feel swapped.
    """
    spectral_t = getattr(development, "spectral_transmittance", None)
    process = str(getattr(development, "color_process", None) or "").lower()
    if spectral_t is not None and process == "c41":
        return _to_rgb_u8(color_negative_lightbox_preview(spectral_t))
    if spectral_t is not None:
        # E-6 developed "film" is already a positive slide.
        return _to_rgb_u8(development.positive_preview)
    return _to_rgb_u8(negative_lightbox_preview(development.transmittance))


_VIEWER_LABELS = {
    "live": "Commit preview (live) — theoretical print",
    "original": "Original photo — click Live print below to swap back",
    "latent": "Latent DN — click Live print below to swap back",
    "negative": "Developed negative — click Live print below to swap back",
}

# The filmstrip holds whichever three stages are *not* in the large preview.
# Clicking a slot swaps it with the preview, so the strip is a stable set of
# three slots whose occupants change.
STRIP_DEFAULT_SLOTS = ("original", "latent", "negative")
STRIP_SHORT_LABELS = {
    "live": "Live print",
    "original": "Original",
    "latent": "Latent DN",
    "negative": "Negative",
}


def _strip_slots(state) -> list[str]:
    slots = (state or {}).get("strip_slots")
    if not slots or len(slots) != 3:
        return list(STRIP_DEFAULT_SLOTS)
    return list(slots)


def _mode_thumb(state, mode, *, live=None, original=None, latent=None, neg=None):
    """Small reference frame for one filmstrip slot."""
    s = state or {}
    if mode == "original":
        return original if original is not None else s.get("original_ref")
    if mode == "latent":
        return latent if latent is not None else s.get("latent_ref")
    if mode == "negative":
        return neg if neg is not None else s.get("neg_ref")
    img = live if live is not None else s.get("live_rgb")
    if img is None:
        return None
    # Only reached while the preview is showing something other than the print.
    return _downscale_rgb(img, REF_MAX_SIDE)


def _strip_updates(state, *, live=None, original=None, latent=None, neg=None):
    """gr.update for each of the three filmstrip slots, in slot order."""
    return [
        gr.update(
            value=_mode_thumb(
                state, mode, live=live, original=original, latent=latent, neg=neg
            ),
            label=STRIP_SHORT_LABELS.get(mode, mode),
        )
        for mode in _strip_slots(state)
    ]


def _viewer_frame(state, live=None, original=None, latent=None, neg=None):
    """Large preview image + label for the current viewer mode."""
    mode = (state or {}).get("viewer_mode", "live")
    if mode == "original":
        img = (state or {}).get("original_view")
        if img is None:
            img = original if original is not None else (state or {}).get("original_ref")
    elif mode == "latent":
        img = (state or {}).get("latent_view")
        if img is None:
            img = latent if latent is not None else (state or {}).get("latent_ref")
    elif mode == "negative":
        img = (state or {}).get("neg_view")
        if img is None:
            img = neg if neg is not None else (state or {}).get("neg_ref")
    else:
        mode = "live"
        img = _display_live_rgb(state, live=live)
    return gr.update(value=img, label=_VIEWER_LABELS.get(mode, _VIEWER_LABELS["live"]))


def _inspect_frame(state, live=None):
    """High-res inspect panel image for zooming the active sequence stage."""
    mode = (state or {}).get("viewer_mode", "live")
    if mode == "original":
        img = (state or {}).get("original_inspect")
        if img is None:
            img = (state or {}).get("original_view")
    elif mode == "latent":
        img = (state or {}).get("latent_inspect")
        if img is None:
            img = (state or {}).get("latent_view")
    elif mode == "negative":
        img = (state or {}).get("neg_inspect")
        if img is None:
            img = (state or {}).get("neg_view")
    else:
        mode = "live"
        img = (state or {}).get("live_inspect")
        if img is None:
            img = live if live is not None else (state or {}).get("live_rgb")
    label = {
        "live": "Inspect — Live theoretical print (scroll-wheel zoom, drag to pan, double-click reset)",
        "original": "Inspect — Original (scroll-wheel zoom, drag to pan, double-click reset)",
        "latent": "Inspect — Latent DN (scroll-wheel zoom, drag to pan, double-click reset)",
        "negative": "Inspect — Developed negative (scroll-wheel zoom, drag to pan, double-click reset)",
    }.get(mode, "Inspect")
    return gr.update(value=img, label=label)


def _pack_preview(live, original, latent, neg, summary, state, *, mark_dirty=False, controls=None):
    status, hist = _split_summary(summary or "")
    if state is not None:
        if live is not None:
            state = {**state, "live_rgb": live, "live_inspect": live}
        if neg is not None:
            # `neg` is the filmstrip-sized reference (REF_MAX_SIDE). Callers
            # that actually recompute the negative store full-size neg_view /
            # neg_inspect on the state first, so only fill those in when they
            # are missing — writing the thumbnail into them made the negative
            # render small in the preview and soft under Inspect zoom.
            updates = {"neg_ref": neg}
            if state.get("neg_view") is None:
                updates["neg_view"] = neg
            if state.get("neg_inspect") is None:
                updates["neg_inspect"] = neg
            state = {**state, **updates}
        if mark_dirty or controls is not None:
            state = _mark_dirty(state, controls)
    shown = _viewer_frame(state, live=live, original=original, latent=latent, neg=neg)
    slot_a, slot_b, slot_c = _strip_updates(
        state, live=live, original=original, latent=latent, neg=neg
    )
    return (
        shown,
        slot_a,
        slot_b,
        slot_c,
        status,
        hist,
        _inspect_frame(state, live=live),
        state,
    )


def swap_strip_slot(index: int):
    """Click a filmstrip slot: it goes to the large preview, and whatever was
    in the preview drops into the slot it came from, so you can toggle back."""

    def _fn(state, evt: SelectData | None = None):
        if not state or state.get("dn") is None:
            empty = gr.update()
            return empty, empty, empty, empty, empty, "*Commit Ingest first.*", state
        slots = _strip_slots(state)
        clicked = slots[index]
        slots[index] = state.get("viewer_mode", "live")
        state = {**state, "viewer_mode": clicked, "strip_slots": slots}
        tip = {
            "live": "_Live theoretical print. Use **Frame** to crop/straighten, **Inspect** to zoom._",
            "original": "_Original in the preview — click it in the strip to swap back._",
            "latent": "_Latent DN in the preview — click it in the strip to swap back._",
            "negative": "_Developed negative in the preview — click it in the strip to swap back._",
        }.get(clicked, "")
        banner = _stage_banner(state.get("stage", "development"), _locks(state))
        status = f"{banner}\n\n{tip}"
        slot_a, slot_b, slot_c = _strip_updates(state)
        return (
            _viewer_frame(state),
            slot_a,
            slot_b,
            slot_c,
            _inspect_frame(state),
            status,
            state,
        )

    return _fn


def on_preview_tool_change(tool: str):
    """Update the live preview label for the active tool mode."""
    tool = str(tool or "print")
    labels = {
        "print": LIVE_PRINT_LABEL + " · easel",
        "frame": LIVE_PRINT_LABEL + " · frame",
        "inspect": LIVE_PRINT_LABEL + " · inspect",
    }
    return gr.update(label=labels.get(tool, LIVE_PRINT_LABEL))


def set_workspace_drawer(name: str):
    """Toggle left drawers — clicking the active icon collapses the drawer."""
    # Implemented primarily in JS; this keeps Gradio accordion open states in sync.
    name = str(name or "ingest")
    return (
        gr.update(open=name == "ingest"),
        gr.update(open=name == "develop"),
        gr.update(open=name == "print"),
        gr.update(open=name == "frame"),
        gr.update(open=name == "log"),
        name,
    )


def _rebuild_views_from_dn(state: dict) -> dict:
    """Refresh latent / proxy caches from the current DN image."""
    dn = state["dn"]
    latent_full = _to_rgb_u8(dn.to_luminance(), assume_linear=True)
    latent_view = _downscale_rgb(latent_full, LIVE_MAX_SIDE)
    latent_inspect = _downscale_rgb(latent_full, INSPECT_MAX_SIDE)
    latent_ref = _downscale_rgb(latent_full, REF_MAX_SIDE)
    return {
        **state,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "latent_view": latent_view,
        "latent_inspect": latent_inspect,
        "latent_ref": latent_ref,
        "live_rgb": latent_view if state.get("development") is None else state.get("live_rgb"),
        "live_inspect": latent_inspect if state.get("development") is None else state.get("live_inspect"),
        "development": None,
        "development_full": None,
        "transmittance_proxy": None,
        "print": None,
        "print_draft": None,
        "neg_ref": None,
        "neg_view": None,
    }


def _unlock_develop_print(dn) -> None:
    ui = dn.metadata.setdefault("ui_state", {})
    locks = ui.setdefault("locked_stages", [])
    committed = ui.setdefault("committed_stages", [])
    for stage in ("print", "development"):
        if stage in locks:
            locks.remove(stage)
        if stage in committed:
            committed.remove(stage)


def _ensure_geometry_bases(state: dict) -> dict:
    """Keep full-res bases for crop/straighten reset (and 90° rotates)."""
    dn = state.get("dn")
    if dn is None:
        return state
    if state.get("geometry_base") is None:
        state = {**state, "geometry_base": np.asarray(dn.image).copy()}
    if state.get("original_base") is None:
        # Prefer the largest original we still have (inspect > live > ref).
        # These are numpy arrays, so `a or b` raises on the truth-value test.
        src = next(
            (
                state.get(key)
                for key in ("original_inspect", "original_view", "original_ref")
                if state.get(key) is not None
            ),
            None,
        )
        if src is not None:
            state = {**state, "original_base": np.asarray(src).copy()}
    return state


def _set_original_previews_from_base(state: dict) -> dict:
    """Rebuild original_* preview sizes from original_base (uint8 RGB)."""
    orig_base = state.get("original_base")
    if orig_base is None:
        return state
    orig_u8 = np.asarray(orig_base)
    if orig_u8.dtype != np.uint8:
        orig_u8 = np.clip(orig_u8, 0, 255).astype(np.uint8)
    state["original_view"] = _downscale_rgb(orig_u8, LIVE_MAX_SIDE)
    state["original_inspect"] = _downscale_rgb(orig_u8, INSPECT_MAX_SIDE)
    state["original_ref"] = _downscale_rgb(orig_u8, REF_MAX_SIDE)
    return state


def rotate_working(turns_cw: int, state):
    """Rotate the Digital Negative and reference previews; clears Develop/Print locks."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")

    state = _ensure_geometry_bases(state)
    dn = state["dn"]
    # 90° redefines the framing base — rotate bases, drop fine straighten/crop.
    if state.get("geometry_base") is not None:
        state["geometry_base"] = rotate_image(state["geometry_base"], turns_cw)
        dn.image = np.asarray(state["geometry_base"]).copy()
    else:
        dn.image = rotate_image(dn.image, turns_cw)

    if state.get("original_base") is not None:
        state["original_base"] = rotate_image(state["original_base"], turns_cw)
        state = _set_original_previews_from_base(state)
    else:
        for key in ("original_view", "original_ref", "original_inspect"):
            if state.get(key) is not None:
                state[key] = rotate_image(state[key], turns_cw)

    ingest = dn.metadata.setdefault("ingest", {})
    degrees = int(ingest.get("rotation_degrees", 0)) + (90 * int(turns_cw))
    ingest["rotation_degrees"] = degrees % 360
    ingest["straighten_degrees"] = 0.0
    ingest["crop"] = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    _unlock_develop_print(dn)
    dn.metadata.setdefault("history", []).append(
        {
            "op": "rotate",
            "degrees_cw": 90 * int(turns_cw),
            "total_degrees": ingest["rotation_degrees"],
        }
    )
    dn.touch()

    state = _rebuild_views_from_dn(
        {
            **state,
            "dn": dn,
            "viewer_mode": "live",
            "strip_slots": list(STRIP_DEFAULT_SLOTS),
            "stage": "development",
            "original_view": state.get("original_view"),
            "original_ref": state.get("original_ref"),
            "original_inspect": state.get("original_inspect"),
            "geometry_base": state.get("geometry_base"),
            "original_base": state.get("original_base"),
        }
    )
    deg = ingest["rotation_degrees"]
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Rotated {90 * int(turns_cw):+d}°** (total {deg}° CW).  \n"
        f"_Develop/Print unlocked — check orientation, then Commit Develop._\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    state["dirty"] = True
    return (
        _viewer_frame(state, live=state.get("live_rgb")),
        state.get("original_ref"),
        state.get("latent_ref"),
        None,
        *_split_summary(summary),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=True),
        gr.update(open=True),
        state,
        0.0,
        DEFAULT_CROP_RECT,
        "free",
    )


def apply_crop_straighten(straighten_deg, crop_rect, crop_ratio, state):
    """Apply fine straighten + interactive crop box from the geometry base."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    state = _ensure_geometry_bases(state)
    base = state.get("geometry_base")
    if base is None:
        raise gr.Error("No framing base — Commit Ingest again.")

    try:
        left, top, right, bottom = parse_crop_rect(crop_rect)
    except ValueError as exc:
        raise gr.Error(f"Invalid crop box: {exc}") from exc
    deg = float(straighten_deg or 0.0)
    ratio = str(crop_ratio or "free")
    if left + right >= 0.92 or top + bottom >= 0.92:
        raise gr.Error("Crop is too aggressive — leave more of the frame.")

    framed = apply_framing(
        base,
        straighten_degrees_cw=deg,
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        fill=0.0,
    )
    if framed.shape[0] < 32 or framed.shape[1] < 32:
        raise gr.Error("Cropped image is too small.")

    dn = state["dn"]
    dn.image = framed
    ingest = dn.metadata.setdefault("ingest", {})
    ingest["straighten_degrees"] = deg
    ingest["crop"] = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "ratio": ratio,
        "rect": str(crop_rect or DEFAULT_CROP_RECT),
    }

    orig_base = state.get("original_base")
    if orig_base is not None:
        orig_framed = apply_framing(
            orig_base.astype(np.float32),
            straighten_degrees_cw=deg,
            crop_left=left,
            crop_top=top,
            crop_right=right,
            crop_bottom=bottom,
            fill=0.0,
        )
        orig_u8 = np.clip(orig_framed, 0, 255).astype(np.uint8)
        state["original_view"] = _downscale_rgb(orig_u8, LIVE_MAX_SIDE)
        state["original_inspect"] = _downscale_rgb(orig_u8, INSPECT_MAX_SIDE)
        state["original_ref"] = _downscale_rgb(orig_u8, REF_MAX_SIDE)

    _unlock_develop_print(dn)
    dn.metadata.setdefault("history", []).append(
        {
            "op": "frame",
            "straighten_degrees": round(deg, 3),
            "crop_left": left,
            "crop_top": top,
            "crop_right": right,
            "crop_bottom": bottom,
            "ratio": ratio,
        }
    )
    dn.touch()

    state = _rebuild_views_from_dn(
        {
            **state,
            "dn": dn,
            "viewer_mode": "live",
            "strip_slots": list(STRIP_DEFAULT_SLOTS),
            "stage": "development",
            "original_view": state.get("original_view"),
            "original_ref": state.get("original_ref"),
            "original_inspect": state.get("original_inspect"),
            "geometry_base": state.get("geometry_base"),
            "original_base": state.get("original_base"),
        }
    )
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Framed** — straighten {deg:+.2f}° · ratio `{ratio}` · "
        f"trim L{left*100:.0f}% T{top*100:.0f}% R{right*100:.0f}% B{bottom*100:.0f}%.  \n"
        f"_Develop/Print unlocked — Commit Develop when the crop looks right._\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    state["dirty"] = True
    rect_echo = str(crop_rect or DEFAULT_CROP_RECT)
    return (
        _viewer_frame(state, live=state.get("live_rgb")),
        state.get("original_ref"),
        state.get("latent_ref"),
        None,
        *_split_summary(summary),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=True),
        gr.update(open=True),
        state,
        deg,
        rect_echo,
        ratio,
    )


def reset_crop_straighten(state):
    """Restore DN / original previews to the geometry base (post-ingest or last 90°)."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    state = _ensure_geometry_bases(state)
    base = state.get("geometry_base")
    if base is None:
        raise gr.Error("No framing base to restore.")

    dn = state["dn"]
    dn.image = np.asarray(base).copy()
    ingest = dn.metadata.setdefault("ingest", {})
    ingest["straighten_degrees"] = 0.0
    ingest["crop"] = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0, "ratio": "free"}

    if state.get("original_base") is not None:
        state = _set_original_previews_from_base(state)

    _unlock_develop_print(dn)
    dn.metadata.setdefault("history", []).append({"op": "frame_reset"})
    dn.touch()

    state = _rebuild_views_from_dn(
        {
            **state,
            "dn": dn,
            "viewer_mode": "live",
            "strip_slots": list(STRIP_DEFAULT_SLOTS),
            "stage": "development",
            "original_view": state.get("original_view"),
            "original_ref": state.get("original_ref"),
            "original_inspect": state.get("original_inspect"),
            "geometry_base": state.get("geometry_base"),
            "original_base": state.get("original_base"),
        }
    )
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Framing reset** — back to the last 90° orientation (no fine straighten/crop).  \n"
        f"_Develop/Print unlocked._\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    return (
        _viewer_frame(state, live=state.get("live_rgb")),
        state.get("original_ref"),
        state.get("latent_ref"),
        None,
        *_split_summary(summary),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=True),
        gr.update(open=True),
        state,
        0.0,
        DEFAULT_CROP_RECT,
        "free",
    )


def suggest_auto_crop(auto_rule, crop_ratio, straighten_deg, state):
    """Set the interactive crop box from classical composition heuristics."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    state = _ensure_geometry_bases(state)
    deg = float(straighten_deg or 0.0)
    # Analyze the same picture the crop stage shows (prefer original photo).
    src = state.get("original_base")
    if src is None:
        src = state.get("geometry_base")
    if src is None:
        raise gr.Error("No framing base to analyze.")
    img = np.asarray(src)
    # Composition heuristics don't need full-res — keep the UI responsive.
    h0, w0 = img.shape[:2]
    max_side = 960
    m = max(h0, w0)
    if m > max_side:
        if img.dtype == np.uint8 or (img.ndim == 3 and img.shape[2] >= 3):
            img = _downscale_rgb(img, max_side)
        else:
            scale = max_side / float(m)
            nh, nw = max(1, int(round(h0 * scale))), max(1, int(round(w0 * scale)))
            yy = (np.linspace(0, h0 - 1, nh)).astype(np.int32)
            xx = (np.linspace(0, w0 - 1, nw)).astype(np.int32)
            img = img[yy][:, xx]
    if abs(deg) >= 1e-6:
        img = straighten_image(
            img.astype(np.float32) if img.dtype == np.uint8 else img,
            deg,
            fill=0.0,
        )
        if np.asarray(src).dtype == np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    image_aspect = w / max(h, 1)
    aspect = parse_aspect_ratio(crop_ratio, image_aspect)
    rule = str(auto_rule or "auto")
    result = suggest_crop_box(img, rule=rule, aspect_ratio=aspect)
    rect = format_crop_rect(result)
    used = AUTO_CROP_RULE_LABELS.get(result.get("rule"), result.get("rule"))
    asked = AUTO_CROP_RULE_LABELS.get(rule, rule)
    hint = (
        f"_Auto crop ready — **{asked}**"
        + (f" → scored as **{used}**" if rule in {"auto", "best"} else "")
        + f" · subject ≈ ({result['subject']['x']:.0%}, {result['subject']['y']:.0%}). "
        f"Tweak the box if needed, then **Apply framing**._"
    )
    return rect, hint


def suggest_auto_straighten(state):
    """Set the straighten slider from detected horizontal structure."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    state = _ensure_geometry_bases(state)
    src = state.get("original_base")
    if src is None:
        src = state.get("geometry_base")
    if src is None:
        raise gr.Error("No framing base to analyze.")
    deg = estimate_straighten_degrees(np.asarray(src))
    if abs(deg) < 0.15:
        hint = "_Auto straighten — already level (0.0°). Adjust the slider if you disagree._"
    else:
        hint = (
            f"_Auto straighten — **{deg:+.2f}°**. "
            f"Tweak if needed, then **Apply framing** (or Auto crop next)._"
        )
    return float(deg), hint


def rotate_cw(state):
    return rotate_working(1, state)


def rotate_ccw(state):
    return rotate_working(-1, state)


def rotate_180(state):
    return rotate_working(2, state)


def _stage_control_updates(state):
    """Button interactivity for the active roll frame."""
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    # Camera roll stays open so more frames can be added later.
    # Clear the file widget so a later "Add to roll" (sample) does not
    # re-ingest the previous upload batch.
    sample_u, file_u, ingest_u = on, gr.update(interactive=True, value=None), on
    if not state or state.get("dn") is None:
        return sample_u, file_u, ingest_u, off, off, off, off, off
    locks = set(_locks(state))
    if "print" in locks:
        return sample_u, file_u, ingest_u, off, on, off, on, on
    if "development" in locks:
        return sample_u, file_u, ingest_u, off, on, on, off, on
    if "ingest" in locks:
        return sample_u, file_u, ingest_u, on, off, off, off, on
    return sample_u, file_u, ingest_u, off, off, off, off, on

def _roll_session_outputs(state, *, drawer: str | None = "roll"):
    """UI tuple shared by add / select / remove on the camera roll."""
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    empty = {"roll": [], "roll_index": -1}
    if not state or state.get("dn") is None:
        summary = (
            "**1. Upload — working** → 2. Develop → 3. Print\n\n"
            "*Add photos from **Upload** — they collect in the **Roll** tab.*"
        )
        return (
            None,
            None,
            None,
            None,
            *_split_summary(summary),
            on,
            on,
            on,
            off,
            off,
            off,
            off,
            gr.update(open=True),
            gr.update(open=True),
            gr.update(open=True),
            off,
            off,
            off,
            off,
            off,
            off,
            off,
            0.0,
            DEFAULT_CROP_RECT,
            "free",
            None,
            empty,
            "ingest",
            _roll_meta_md(empty),
            _roll_gallery_update(empty),
            off,
        )

    summary = state.get("summary_cache") or ""
    n = len(state.get("roll") or [])
    idx = int(state.get("roll_index", 0)) + 1
    if n > 1 and "**Upload locked**" in summary:
        summary = summary.replace(
            "**Upload locked**",
            f"**Upload locked** (frame {idx}/{n})",
            1,
        )
    elif n > 1 and "**Ingest locked**" in summary:
        summary = summary.replace(
            "**Ingest locked**",
            f"**Upload locked** (frame {idx}/{n})",
            1,
        )
    elif n > 1 and "frame " not in summary.split("\n", 1)[0]:
        banner_note = f"_Roll · frame {idx}/{n}_\n\n"
        if banner_note.strip() not in summary:
            summary = summary.replace("\n\n", f"\n\n{banner_note}", 1)

    (
        sample_u,
        file_u,
        ingest_u,
        develop_u,
        unlock_dev_u,
        print_u,
        unlock_print_u,
        remove_u,
    ) = _stage_control_updates(state)

    live = state.get("live_rgb")
    if live is None:
        live = state.get("latent_view")
    inspect_live = state.get("live_inspect")
    if inspect_live is None:
        inspect_live = state.get("latent_inspect")
    mode = state.get("viewer_mode", "live")
    active = drawer if drawer is not None else _drawer_for_frame(state)
    return (
        gr.update(value=live, label=_VIEWER_LABELS.get(mode, _VIEWER_LABELS["live"])),
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        *_split_summary(summary),
        sample_u,
        file_u,
        ingest_u,
        develop_u,
        print_u,
        unlock_dev_u,
        unlock_print_u,
        gr.update(open=True),
        gr.update(open=True),
        gr.update(open=True),
        on,
        on,
        on,
        on,
        on,
        on,
        on,
        0.0,
        DEFAULT_CROP_RECT,
        "free",
        _inspect_frame(state, live=inspect_live),
        state,
        active,
        _roll_meta_md(state),
        _roll_gallery_update(state),
        remove_u,
    )

def commit_ingest(sample_path, file_obj, state):
    """Append one or more photos to the camera roll and activate the last one."""
    paths = _collect_input_paths(file_obj, sample_path)
    if not paths:
        raise gr.Error("Choose a sample or upload one or more photos.")

    state = _ensure_roll(state)
    if state.get("dn") is not None and state.get("roll"):
        state = _sync_active_into_roll(state)

    roll = list(state.get("roll") or [])
    for path in paths:
        roll.append(_build_ingest_frame(path))

    new_index = len(roll) - 1
    state = {**roll[new_index], "roll": roll, "roll_index": new_index}
    n = len(roll)
    added = len(paths)
    summary = state["summary_cache"]
    if n > 1 or added > 1:
        summary = (
            f"{_stage_banner('development', ['ingest'])}\n\n"
            f"**Added {added} to Roll** — frame {new_index + 1}/{n} active  \n"
            f"`{state['dn'].metadata['source']['original_filename']}`  \n"
            f"_Open the **Roll** tab to switch or remove frames._\n\n"
            f"{_history_md(state['dn'])}"
        )
        state["summary_cache"] = summary
        roll[new_index] = _frame_payload(state)
        state["roll"] = roll
    return _session_with_controls(state, drawer="roll")


def _roll_switch_bundle(state, *, drawer="roll", modal_visible=False, pending=-1, restore_controls=True):
    """Session UI + control restore + save-prompt modal for frame switches."""
    base = _roll_session_outputs(state, drawer=drawer)
    if restore_controls:
        ctrls = _control_updates(state)
    elif state and state.get("dn") is not None:
        # Stay on this frame's widget values (dirty prompt); only sync lock flags.
        ctrls = _control_interactivity_updates(state)
    else:
        ctrls = tuple(gr.skip() for _ in range(_CONTROL_COUNT))
    return (
        *base,
        *ctrls,
        gr.update(visible=bool(modal_visible)),
        int(pending),
    )


def begin_roll_switch(index_raw, state, *control_args):
    """Start a frame switch — prompt when the current frame has unsaved work.

    control_args snapshots the outgoing frame's Develop/Print UI so settings
    stay per-frame instead of leaking through shared Gradio widgets.
    """
    # Careful: index 0 is valid — never use `index_raw or ""`.
    if index_raw is None:
        raw = ""
    else:
        raw = str(index_raw).strip()
    # Ignore empty/-1 from the hidden textbox mounting or resets.
    if raw == "" or raw == "-1":
        if not state or state.get("dn") is None:
            return _roll_switch_bundle(None, modal_visible=False, pending=-1)
        return _roll_switch_bundle(
            state, drawer="roll", modal_visible=False, pending=-1, restore_controls=False
        )
    if not state or not state.get("roll"):
        return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)
    target = _parse_roll_index(raw, fallback=-1)
    current = int(state.get("roll_index", -1))
    if target < 0 or target >= len(state.get("roll") or []):
        return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)
    if target == current:
        return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1, restore_controls=False)

    # Stash the live UI onto the outgoing frame before any switch/prompt.
    state = _attach_controls(state, *control_args)

    if _is_dirty(state):
        # Keep working on the current frame until the user chooses Save / Discard.
        return _roll_switch_bundle(
            state,
            drawer="roll",
            modal_visible=True,
            pending=target,
            restore_controls=False,
        )

    state = _activate_roll_index(state, target, save_current=True)
    return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)


def select_roll_frame(state, evt: SelectData | None = None, *control_args):
    """Gallery select → same save-aware switch path as an explicit thumb click."""
    index = evt.index if evt is not None else (state or {}).get("roll_index", 0)
    if isinstance(index, (list, tuple)):
        index = index[0]
    return begin_roll_switch(index, state, *control_args)


def save_and_switch_roll(pending, state, *control_args):
    """Save the active frame into the roll, then activate the pending frame."""
    if not state or not state.get("roll"):
        return _roll_switch_bundle(None, modal_visible=False, pending=-1)
    try:
        target = int(pending)
    except (TypeError, ValueError):
        return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)
    state = _attach_controls(state, *control_args)
    if len(control_args) >= _CONTROL_COUNT:
        state = {**state, "dirty": True}
    state = _activate_roll_index(state, target, save_current=True)
    return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)


def discard_and_switch_roll(pending, state):
    """Abandon unsaved work on the active frame and activate the pending frame."""
    if not state or not state.get("roll"):
        return _roll_switch_bundle(None, modal_visible=False, pending=-1)
    try:
        target = int(pending)
    except (TypeError, ValueError):
        return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)
    state = _activate_roll_index(state, target, save_current=False)
    return _roll_switch_bundle(state, drawer="roll", modal_visible=False, pending=-1)


def cancel_roll_switch(state):
    """Dismiss the save prompt and stay on the current frame."""
    return _roll_switch_bundle(
        state,
        drawer="roll",
        modal_visible=False,
        pending=-1,
        restore_controls=False,
    )


def _parse_roll_index(index_raw, fallback: int = -1) -> int:
    """Accept plain ints or 'index:token' from the hover-✕ UI script."""
    raw = str(index_raw or "").strip()
    if not raw or raw == "-1":
        return int(fallback)
    head = raw.split(":", 1)[0].strip()
    try:
        return int(head)
    except (TypeError, ValueError):
        return int(fallback)


def remove_from_roll(index_raw, state):
    """Drop a frame from the camera roll (hover ✕ passes the index)."""
    raw = str(index_raw or "").strip()
    # Ignore empty/-1 from the hidden textbox mounting or resets — never
    # fall back to "delete whatever is active" on a blank change event.
    if not raw or raw == "-1":
        if not state or state.get("dn") is None:
            return _session_with_controls(None)
        return _session_with_controls(state, drawer="roll")

    state = _ensure_roll(state or {})
    if state.get("dn") is not None and state.get("roll"):
        state = _sync_active_into_roll(state)
    roll = list(state.get("roll") or [])
    idx = _parse_roll_index(raw, fallback=-1)
    if not roll or idx < 0 or idx >= len(roll):
        if not roll or state.get("dn") is None:
            return _session_with_controls(None)
        return _session_with_controls(state, drawer="roll")

    active = int(state.get("roll_index", 0))
    roll.pop(idx)
    if not roll:
        return _session_with_controls(None)
    if active == idx:
        new_idx = min(idx, len(roll) - 1)
    elif active > idx:
        new_idx = active - 1
    else:
        new_idx = active
    state = {**roll[new_idx], "roll": roll, "roll_index": new_idx}
    n = len(roll)
    summary = (
        f"{_stage_banner(state.get('stage', 'development'), _locks(state))}\n\n"
        f"**Removed from roll** — frame {new_idx + 1}/{n} active  \n"
        f"`{state['dn'].metadata['source']['original_filename']}`\n\n"
        f"{_history_md(state['dn'])}"
    )
    state["summary_cache"] = summary
    roll[new_idx] = _frame_payload(state)
    state["roll"] = roll
    return _session_with_controls(state, drawer="roll")


def _print_transmittance(development) -> np.ndarray:
    """Prefer spectral transmittance for color develops."""
    if development is None:
        return None
    spectral = getattr(development, "spectral_transmittance", None)
    if spectral is not None:
        return spectral
    return development.transmittance


def _stash_color_filtration(dn, cc_cyan, cc_magenta, cc_yellow):
    print_meta = dn.metadata.setdefault("print", {})
    print_meta["cc_cyan"] = float(cc_cyan)
    print_meta["cc_magenta"] = float(cc_magenta)
    print_meta["cc_yellow"] = float(cc_yellow)
    filt = print_meta.setdefault("filtration", {})
    filt["type"] = "color" if any(float(x) for x in (cc_cyan, cc_magenta, cc_yellow)) else filt.get(
        "type", "multigrade"
    )
    values = dict(filt.get("values") or {})
    values.update(
        {
            "cc_cyan": float(cc_cyan),
            "cc_magenta": float(cc_magenta),
            "cc_yellow": float(cc_yellow),
        }
    )
    filt["values"] = values


def _run_live_develop_then_print(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    state,
    *,
    max_side: int = LIVE_MAX_SIDE,
    chemistry_mode: str = "bw",
    cc_cyan: float = 0.0,
    cc_magenta: float = 0.0,
    cc_yellow: float = 0.0,
):
    """Develop with current settings, then print with current/default print settings.

    Large viewer shows the theoretical final print — what you'd get after
    Commit Develop + Commit Print with these controls.
    """
    if max_side <= DRAG_MAX_SIDE:
        proxy = state.get("proxy_drag") or _proxy_dn(state["dn"], DRAG_MAX_SIDE)
    else:
        proxy = state.get("proxy") or _proxy_dn(state["dn"], LIVE_MAX_SIDE)
    proxy.metadata["process_seed"] = state["dn"].metadata.get("process_seed")
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        proxy,
        profile,
        development_minutes=float(development_minutes),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
        process_variation=1.0,
        exposure_index=float(exposure_index),
        contrast_filter=str(contrast_filter),
        scene_exposure_seconds=float(scene_exposure),
        halation=float(halation),
        commit=False,
    )
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    tech = _technique_kwargs(
        split_on, soft_grade, hard_grade, soft_seconds, hard_seconds,
        test_strips_on, test_bands, test_stops, flash_stops, dry_down, tone, border_frac,
    )
    t_print = _print_transmittance(development)
    if getattr(development, "color_process", None):
        _stash_color_filtration(state["dn"], cc_cyan, cc_magenta, cc_yellow)
    printed = print_negative(
        t_print,
        state["dn"],
        paper,
        base_exposure_seconds=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        local_stops=local_stops_from_state(state),
        commit=False,
        **tech,
    )
    live_rgb = _to_rgb_u8(printed.preview)
    neg_full = _color_or_bw_negative_view(development)
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    speed = state["dn"].metadata.get("print", {}).get("filtration", {}).get("values", {}).get(
        "filter_speed", 1.0
    )
    quality_note = "drag" if max_side <= DRAG_MAX_SIDE else "hq"
    curve_src = proxy.metadata.get("development", {}).get("curve_source", "?")
    process = getattr(development, "color_process", None) or "bw"
    mode_note = f" · {process.upper()}" if process != "bw" else ""
    if process in {"c41", "e6"}:
        paper_line = (
            f"{paper.name} · CC C{float(cc_cyan):.0f}/M{float(cc_magenta):.0f}/Y{float(cc_yellow):.0f} · "
            f"{_print_timer_label(print_exposure)}"
            if process == "c41"
            else "E-6 slide finish"
        )
    else:
        paper_line = (
            f"{paper.name} · g{float(print_grade):.1f} · {_print_timer_label(print_exposure)} "
            f"· ×{float(speed):.2f}"
        )
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Live print** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note}){mode_note}  \n"
        f"{profile.name} · {developer_id} · {float(development_minutes):g} min · "
        f"curve={curve_src} · N±={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
        f"{paper_line}\n\n"
        f"{_history_md(state['dn'])}"
    )
    state = {
        **state,
        "proxy": state.get("proxy") or _proxy_dn(state["dn"], LIVE_MAX_SIDE),
        "proxy_drag": state.get("proxy_drag") or _proxy_dn(state["dn"], DRAG_MAX_SIDE),
        "development": development,
        "spectral_transmittance": getattr(development, "spectral_transmittance", None),
        "chemistry_mode": str(chemistry_mode or "bw"),
        "live_rgb": live_rgb,
        "live_inspect": live_rgb,
        "neg_ref": neg_ref,
        "neg_view": neg_view,
        "neg_inspect": neg_inspect,
        "stage": "development",
        "summary_cache": summary,
        "draft_print": {
            "paper_id": paper_id,
            "print_exposure": float(print_exposure),
            "print_grade": float(print_grade),
            "print_contrast": float(print_contrast),
            "cc_cyan": float(cc_cyan),
            "cc_magenta": float(cc_magenta),
            "cc_yellow": float(cc_yellow),
        },
        "print_technique": tech,
    }
    state = _remember_print_seconds(state, print_exposure)
    return live_rgb, neg_ref, summary, state


def live_preview(
    chemistry_mode,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    cc_cyan,
    cc_magenta,
    cc_yellow,
    state,
    quality: str = "high",
    mark_dirty: bool = False,
):
    """Unified live viewer: develop+print while developing; print-only after Develop lock.

    quality='drag' uses a faster lower-res proxy while sliders move;
    quality='high' (release / change) uses commit-accurate resolution.
    mark_dirty=True when the user edited controls (not after a roll switch restore).
    """
    max_side = DRAG_MAX_SIDE if quality == "drag" else LIVE_MAX_SIDE
    controls = _capture_controls(
        chemistry_mode,
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        split_on,
        soft_grade,
        hard_grade,
        soft_seconds,
        hard_seconds,
        test_strips_on,
        test_bands,
        test_stops,
        flash_stops,
        dry_down,
        tone,
        border_frac,
        cc_cyan,
        cc_magenta,
        cc_yellow,
    )

    if not state or state.get("dn") is None:
        return _pack_preview(None, None, None, None, "*Commit Ingest first.*", state)

    if _locked(state, "print"):
        return _pack_preview(
            state.get("live_rgb"),
            state.get("original_ref"),
            state.get("latent_ref"),
            state.get("neg_ref"),
            state.get("summary_cache", ""),
            state,
            mark_dirty=mark_dirty,
            controls=controls if mark_dirty else None,
        )

    if _locked(state, "development"):
        # Print-only commit preview
        if state.get("development_full") is not None:
            t = state["development_full"].transmittance
            step = max(1, int(np.ceil(max(t.shape) / max_side)))
            t = np.ascontiguousarray(t[::step, ::step])
        else:
            t = state.get("transmittance_proxy")
            if t is None and state.get("development") is not None:
                t = state["development"].transmittance
            if t is None:
                return _pack_preview(
                    state.get("live_rgb"),
                    state.get("original_ref"),
                    state.get("latent_ref"),
                    state.get("neg_ref"),
                    state.get("summary_cache", ""),
                    state,
                    mark_dirty=mark_dirty,
                    controls=controls if mark_dirty else None,
                )
            if max(t.shape) > max_side:
                step = max(1, int(np.ceil(max(t.shape) / max_side)))
                t = np.ascontiguousarray(t[::step, ::step])

        paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
        tech = _technique_kwargs(
            split_on, soft_grade, hard_grade, soft_seconds, hard_seconds,
            test_strips_on, test_bands, test_stops, flash_stops, dry_down, tone, border_frac,
        )
        dev_full = state.get("development_full") or state.get("development")
        t_use = t
        if dev_full is not None and getattr(dev_full, "spectral_transmittance", None) is not None:
            t_use = dev_full.spectral_transmittance
            if max(t_use.shape[:2]) > max_side:
                step = max(1, int(np.ceil(max(t_use.shape[:2]) / max_side)))
                t_use = np.ascontiguousarray(t_use[::step, ::step, ...])
            _stash_color_filtration(state["dn"], cc_cyan, cc_magenta, cc_yellow)
        result = print_negative(
            t_use,
            state["dn"],
            paper,
            base_exposure_seconds=float(print_exposure),
            grade=float(print_grade),
            contrast=float(print_contrast),
            local_stops=local_stops_from_state(state),
            commit=False,
            **tech,
        )
        live_rgb = _to_rgb_u8(result.preview)
        speed = state["dn"].metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
        quality_note = "drag" if quality == "drag" else "hq"
        strokes = state.get("db_strokes") or []
        db_note = f" · dodge/burn ×{len(strokes)}" if strokes else ""
        exposing = " · **EXPOSING**" if state.get("db_exposing") else ""
        summary = (
            f"{_stage_banner('print', _locks(state))}\n\n"
            f"**Print preview** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note})  \n"
            f"{paper.name} · g{float(print_grade):.1f} · {_print_timer_label(print_exposure)} · "
            f"×{float(speed):.2f}{db_note}{exposing}\n\n{_history_md(state['dn'])}"
        )
        state = {
            **state,
            "print_draft": result,
            "live_rgb": live_rgb,
            "summary_cache": summary,
            "print_technique": tech,
        }
        state = _remember_print_seconds(state, print_exposure)
        return _pack_preview(
            live_rgb,
            state.get("original_ref"),
            state.get("latent_ref"),
            state.get("neg_ref"),
            summary,
            state,
            mark_dirty=mark_dirty,
            controls=controls if mark_dirty else None,
        )

    # Develop unlocked: show print through the working negative
    live_rgb, neg_ref, summary, state = _run_live_develop_then_print(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        split_on,
        soft_grade,
        hard_grade,
        soft_seconds,
        hard_seconds,
        test_strips_on,
        test_bands,
        test_stops,
        flash_stops,
        dry_down,
        tone,
        border_frac,
        state,
        max_side=max_side,
        chemistry_mode=str(chemistry_mode or "bw"),
        cc_cyan=float(cc_cyan),
        cc_magenta=float(cc_magenta),
        cc_yellow=float(cc_yellow),
    )
    return _pack_preview(
        live_rgb,
        state.get("original_ref"),
        state.get("latent_ref"),
        neg_ref,
        summary,
        state,
        mark_dirty=mark_dirty,
        controls=controls if mark_dirty else None,
    )


def live_preview_drag(
    chemistry_mode,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    cc_cyan,
    cc_magenta,
    cc_yellow,
    state,
):
    return live_preview(
        chemistry_mode,
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        split_on,
        soft_grade,
        hard_grade,
        soft_seconds,
        hard_seconds,
        test_strips_on,
        test_bands,
        test_stops,
        flash_stops,
        dry_down,
        tone,
        border_frac,
        cc_cyan,
        cc_magenta,
        cc_yellow,
        state,
        quality="drag",
        mark_dirty=True,
    )



def live_preview_high(
    chemistry_mode,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    cc_cyan,
    cc_magenta,
    cc_yellow,
    state,
):
    return live_preview(
        chemistry_mode,
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        split_on,
        soft_grade,
        hard_grade,
        soft_seconds,
        hard_seconds,
        test_strips_on,
        test_bands,
        test_stops,
        flash_stops,
        dry_down,
        tone,
        border_frac,
        cc_cyan,
        cc_magenta,
        cc_yellow,
        state,
        quality="high",
        mark_dirty=False,
    )



def live_preview_edit(
    chemistry_mode,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
    cc_cyan,
    cc_magenta,
    cc_yellow,
    state,
):
    return live_preview(
        chemistry_mode,
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        split_on,
        soft_grade,
        hard_grade,
        soft_seconds,
        hard_seconds,
        test_strips_on,
        test_bands,
        test_stops,
        flash_stops,
        dry_down,
        tone,
        border_frac,
        cc_cyan,
        cc_magenta,
        cc_yellow,
        state,
        quality="high",
        mark_dirty=True,
    )



def commit_develop(
    film_id, developer_id, development_minutes, contrast, grain,
    exposure_index, contrast_filter, scene_exposure, halation, state,
):
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")
    if _locked(state, "development"):
        raise gr.Error("Develop is already locked.")

    dn = state["dn"]
    profile = load_film_profile(_profile_path(list_film_profiles(), film_id))
    development = develop(
        dn,
        profile,
        development_minutes=float(development_minutes),
        contrast_modifier=float(contrast),
        grain_strength=float(grain),
        developer_id=developer_id,
        process_variation=1.0,
        exposure_index=float(exposure_index),
        contrast_filter=str(contrast_filter),
        scene_exposure_seconds=float(scene_exposure),
        halation=float(halation),
        commit=True,
    )
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "development" not in locks:
        locks.append("development")

    t_src = _print_transmittance(development)
    if t_src.ndim >= 2:
        step = max(1, int(np.ceil(max(t_src.shape[:2]) / LIVE_MAX_SIDE)))
        t_proxy = np.ascontiguousarray(t_src[::step, ::step, ...])
    else:
        t_proxy = t_src

    # Keep last theoretical print on screen until .then refreshes with Print controls;
    # fall back to positive if no live print was generated yet.
    # Do not use `or` — live_rgb is a numpy array (ambiguous truth value).
    live_view = state.get("live_rgb")
    if live_view is None:
        live_view = _downscale_rgb(
            _to_rgb_u8(development.positive_preview), LIVE_MAX_SIDE
        )
    neg_full = _color_or_bw_negative_view(development)
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    process = getattr(development, "color_process", None)
    process_note = f" ({process.upper()})" if process else ""
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Develop locked**{process_note} — refine Print below, then Commit Print.\n\n{_history_md(dn)}"
    )
    state = {
        **state,
        "dn": dn,
        "proxy": state.get("proxy"),
        "proxy_drag": state.get("proxy_drag"),
        "original_ref": state.get("original_ref"),
        "original_view": state.get("original_view"),
        "original_inspect": state.get("original_inspect"),
        "latent_ref": state.get("latent_ref"),
        "latent_view": state.get("latent_view"),
        "latent_inspect": state.get("latent_inspect"),
        "neg_ref": neg_ref,
        "neg_view": neg_view,
        "neg_inspect": neg_inspect,
        "live_rgb": live_view,
        "live_inspect": live_view,
        "viewer_mode": state.get("viewer_mode", "live"),
        "development": development,
        "development_full": development,
        "transmittance_proxy": t_proxy,
        "spectral_transmittance": getattr(development, "spectral_transmittance", None),
        "stage": "print",
        "summary_cache": summary,
        "source_path": state.get("source_path"),
        "db_accum": None,
        "db_exposing": False,
        "db_seconds_left": 0,
        "db_strokes": [],
    }
    neg_download = _write_negative_package(neg_full, dn)
    state["dl_negative"] = neg_download
    # Keep the camera-roll snapshot aligned with the active frame.
    state = _sync_active_into_roll(state)
    return (
        # Only the negative exists yet, so the trigger downloads it directly
        # rather than opening a menu with one entry.
        gr.update(value="⇣ Download negative", visible=True),
        "negative",
        gr.update(value=neg_download),
        _viewer_frame(state, live=live_view, neg=neg_view),
        state.get("original_ref"),
        state.get("latent_ref"),
        neg_ref,
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(open=False),  # collapse Develop
        gr.update(open=True),   # show Print
        state,
        "print",
    )


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(text)).strip("-")


def _source_stem(dn) -> str:
    source = dn.metadata.get("source", {}).get("original_filename") or "frame"
    return Path(str(source)).stem or "frame"


def _downloads_dir() -> Path:
    out = Path(tempfile.gettempdir()) / "darkroom_downloads"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_png(rgb, path: Path) -> Path:
    # optimize=True buys almost nothing on photographic data and costs time on
    # a full-res frame, so stay on the default compression.
    from PIL import Image

    Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(path, format="PNG")
    return path


def _zip_package(members: list[tuple[str, Path]], zip_name: str) -> str:
    """Bundle files as <folder>/<file> inside a zip, so they unpack into
    print/ and negative/ directories rather than loose images."""
    import zipfile

    path = _downloads_dir() / zip_name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, src in members:
            zf.write(src, arcname=arcname)
    return str(path)


def _negative_png(neg_rgb, dn) -> Path:
    stem = _source_stem(dn)
    return _save_png(neg_rgb, _downloads_dir() / f"{stem}_negative.png")


def _write_negative_package(neg_rgb, dn) -> str:
    """Negative on its own — unpacks into negative/."""
    stem = _source_stem(dn)
    png = _negative_png(neg_rgb, dn)
    return _zip_package([(f"negative/{png.name}", png)], f"{stem}_negative.zip")


def _write_print_packages(print_rgb, dn, paper, grade, exposure) -> tuple[str, str]:
    """(print-only, print+negative) zips. Print unpacks into print/, the
    combined package into print/ and negative/ side by side."""
    stem = _source_stem(dn)
    recipe = f"{_safe_name(paper.name)}_g{float(grade):.1f}_{float(exposure):g}s"
    print_png = _save_png(print_rgb, _downloads_dir() / f"{stem}__{recipe}.png")

    print_only = _zip_package(
        [(f"print/{print_png.name}", print_png)], f"{stem}__{recipe}_print.zip"
    )

    members = [(f"print/{print_png.name}", print_png)]
    neg_png = _downloads_dir() / f"{stem}_negative.png"
    if neg_png.exists():
        members.append((f"negative/{neg_png.name}", neg_png))
    combined = _zip_package(members, f"{stem}__{recipe}_print+negative.zip")
    return print_only, combined


def _technique_kwargs(
    split_on,
    soft_grade,
    hard_grade,
    soft_seconds,
    hard_seconds,
    test_strips_on,
    test_bands,
    test_stops,
    flash_stops,
    dry_down,
    tone,
    border_frac,
) -> dict:
    """Pack Print-technique controls for :func:`print_negative`."""
    return {
        "split_grade": bool(split_on),
        "soft_grade": float(soft_grade),
        "hard_grade": float(hard_grade),
        "soft_exposure_seconds": float(soft_seconds),
        "hard_exposure_seconds": float(hard_seconds),
        "test_strips": bool(test_strips_on),
        "test_strip_bands": int(test_bands),
        "test_strip_stops": float(test_stops),
        "flash_stops": float(flash_stops),
        "dry_down_percent": float(dry_down),
        "tone": str(tone or "none"),
        "border_frac": float(border_frac),
    }


def _technique_from_state(state) -> dict:
    tech = (state or {}).get("print_technique")
    return dict(tech) if isinstance(tech, dict) else {}


def commit_print(paper_id, print_exposure, print_grade, print_contrast, state):
    if not state or state.get("development_full") is None:
        raise gr.Error("Commit Develop first.")
    if _locked(state, "print"):
        raise gr.Error("Print is already locked.")
    if state.get("db_exposing"):
        raise gr.Error("Wait for the dodge/burn timer to finish.")

    dn = state["dn"]
    development = state["development_full"]
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    strokes = list(state.get("db_strokes") or [])
    dn.metadata.setdefault("print", {})["dodge_burn"] = strokes
    result = print_negative(
        development.transmittance,
        dn,
        paper,
        base_exposure_seconds=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        local_stops=local_stops_from_state(state),
        commit=True,
        **_technique_from_state(state),
    )
    stages = dn.metadata["ui_state"].setdefault("committed_stages", [])
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "print" not in stages:
        stages.append("print")
    if "print" not in locks:
        locks.append("print")

    print_full = _to_rgb_u8(result.preview)
    live_rgb = _downscale_rgb(print_full, LIVE_MAX_SIDE)
    speed = dn.metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
    db_note = f" · {len(strokes)} dodge/burn pass(es)" if strokes else ""
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Print locked** — {paper.name} · g{float(print_grade):.1f} · "
        f"{_print_timer_label(print_exposure)}{db_note}\n\n{_history_md(dn)}"
    )
    print_only, print_plus_neg = _write_print_packages(
        print_full, dn, paper, print_grade, print_exposure
    )
    state = {
        **state,
        "print": result,
        "live_rgb": live_rgb,
        "summary_cache": summary,
        "dl_print_only": print_only,
        "dl_print_negative": print_plus_neg,
    }
    state = _sync_active_into_roll(state)
    return (
        live_rgb,
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(value="⇣ Download…", visible=True),
        "print,both,negative",
        gr.update(value=print_only),
        gr.update(value=print_plus_neg),
        state,
    )


def _unlock_stage(dn, stage: str) -> None:
    ui = dn.metadata.setdefault("ui_state", {})
    locks = ui.setdefault("locked_stages", [])
    committed = ui.setdefault("committed_stages", [])
    if stage in locks:
        locks.remove(stage)
    if stage in committed and stage != "ingest":
        committed.remove(stage)
    dn.metadata.setdefault("history", []).append({"op": "unlock", "stage": stage})
    dn.touch()



def _print_timer_label(print_seconds) -> str:
    """Human label for the enlarger base timer (seconds + calibrated stops)."""
    seconds = float(print_seconds)
    stops = base_seconds_to_stops(seconds)
    return f"{seconds:g}s base (≈ {stops:+.2f} stops)"


def _pass_math_md(base_seconds, pass_seconds, mode) -> str:
    """Light darkroom math under the dodge/burn timer."""
    base = max(float(base_seconds), 1e-6)
    sec = max(float(pass_seconds), 0.0)
    mode_key = "dodge" if str(mode).lower().startswith("dodge") else "burn"
    stops = relative_pass_stops(sec, base, mode_key)
    if mode_key == "dodge":
        return (
            f"**Pass math** — dodge **{sec:g}s** of **{base:g}s** base "
            f"≈ **{stops:+.2f} stops** if the card is held still "
            f"(holds back light → **lighter** print; area gets "
            f"{max(base - min(sec, base * 0.95), base * 0.05):.1f}s of enlarger light)."
        )
    return (
        f"**Pass math** — burn **{sec:g}s** on **{base:g}s** base "
        f"≈ **{stops:+.2f} stops** if held still "
        f"(adds enlarger light → **darker** print; area gets {base + sec:.1f}s of light)."
    )


def _base_math_md(base_seconds) -> str:
    seconds = float(base_seconds)
    stops = base_seconds_to_stops(seconds)
    return (
        f"**Base timer** — **{seconds:g}s** enlarger clock "
        f"(≈ **{stops:+.2f} stops** vs calibrated {REFERENCE_BASE_SECONDS:g}s). "
        f"Dodge/burn passes are timed against this."
    )


def _remember_print_seconds(state, print_seconds):
    """Keep base timer on state so dodge/burn converts light-time correctly."""
    if not state:
        return state
    seconds = float(print_seconds)
    return {
        **state,
        "print_base_seconds": seconds,
        "db_base_seconds": seconds,
    }


def _db_target_shape(state) -> tuple[int, int]:
    """Height, width for the dodge/burn accumulation map."""
    t = state.get("transmittance_proxy")
    if t is None and state.get("development_full") is not None:
        full = state["development_full"].transmittance
        step = max(1, int(np.ceil(max(full.shape) / LIVE_MAX_SIDE)))
        t = full[::step, ::step]
    if t is None and state.get("development") is not None:
        t = state["development"].transmittance
    if t is None:
        return (512, 512)
    return int(t.shape[0]), int(t.shape[1])


def _editor_from_print(rgb: np.ndarray | None) -> dict:
    """Seed a dark tool-workshop canvas sized like the current print."""
    if rgb is not None and getattr(rgb, "shape", None) is not None and len(rgb.shape) >= 2:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        side = 480
        if h >= w:
            hh, ww = side, max(64, int(round(side * w / max(h, 1))))
        else:
            ww, hh = side, max(64, int(round(side * h / max(w, 1))))
        return tool_workshop_canvas(hh, ww)
    return tool_workshop_canvas()


def _db_flag_html(state) -> str:
    exposing = "1" if state and state.get("db_exposing") else "0"
    stamp = ""
    frac = "0.28"
    shape = "soft_oval"
    mode = "burn"
    if state and state.get("db_stamp_url"):
        stamp = str(state["db_stamp_url"])
    if state and state.get("db_stamp_frac"):
        frac = f"{float(state['db_stamp_frac']):.4f}"
    if state and state.get("db_shape"):
        shape = str(state["db_shape"])
    if state and state.get("db_mode"):
        mode = str(state["db_mode"])
    # Put stamp in an <img> (not a huge data-* attribute) so Gradio keeps it intact.
    img = f'<img id="db_stamp_asset" src="{stamp}" alt="" />' if stamp else ""
    return (
        f'<div data-exposing="{exposing}" data-shape="{shape}" data-stamp-fw="{frac}" '
        f'data-mode="{mode}">{img}</div>'
    )


def _sync_tool_preview(shape_id, mode, state):
    """Keep flag HTML in sync with card shape / mode so the outline can preview before Start."""
    state = {**(state or {})}
    state["db_shape"] = str(shape_id or state.get("db_shape") or "soft_oval")
    state["db_mode"] = (
        "dodge" if str(mode).lower().startswith("dodge") else "burn"
    )
    return state, _db_flag_html(state)


def _wave_banner_html(state) -> str:
    """Big cue above the live print while the enlarger card is active."""
    if not state or not state.get("db_exposing"):
        return '<div class="db-wave-idle"></div>'
    mode = str(state.get("db_mode", "burn"))
    verb = "DODGE" if mode == "dodge" else "BURN"
    left = max(0, int(np.ceil(float(state.get("db_seconds_left", 0.0)))))
    tip = "hold back light → lighter print" if mode == "dodge" else "add enlarger light → darker print"
    return (
        f'<div class="db-wave-active" role="status">'
        f'<span class="db-wave-arrow">↓</span>'
        f'<span><strong>{verb}</strong> — tool outline follows your pointer on the print '
        f'({tip}) · <strong>{left}s</strong> left</span>'
        f'<span class="db-wave-arrow">↓</span>'
        f"</div>"
    )


LIVE_PRINT_LABEL = "Live print — theoretical enlarger print"
LIVE_WAVE_LABEL = "→ WAVE YOUR CARD OVER THIS PRINT ←"


def start_dodge_burn(
    mode, seconds, shape_id, paper_id, print_exposure, print_grade, print_contrast, editor, db_pos, state
):
    if not state or state.get("development_full") is None:
        raise gr.Error("Commit Develop first — dodge/burn happens on the print.")
    if _locked(state, "print"):
        raise gr.Error("Unlock Print to dodge/burn.")
    if state.get("db_exposing"):
        raise gr.Error("Already exposing — wait for the timer.")

    seconds = int(round(float(seconds)))
    if seconds < 1:
        raise gr.Error("Timer must be at least 1 second.")

    h, w = _db_target_shape(state)
    stamp = resolve_tool_stamp(
        shape_id, editor, target_height=h, target_width=w
    )
    if stamp is None or float(stamp.max()) < 0.12:
        raise gr.Error(
            "Pick a card shape (oval/circle/finger) or paint a custom shape, then Start."
        )

    mode = "dodge" if str(mode).lower().startswith("dodge") else "burn"
    base_seconds = float(print_exposure)
    if mode == "dodge" and seconds > base_seconds:
        raise gr.Error(
            f"Dodge pass ({seconds}s) can’t exceed the base exposure ({base_seconds:g}s). "
            "Shorten the dodge timer or lengthen the base."
        )

    parsed = parse_pointer_state(db_pos)
    tool_scale = float(parsed[2]) if parsed is not None else 1.0
    seed_nx, seed_ny = (0.5, 0.5) if parsed is None else (parsed[0], parsed[1])

    state = {**state}
    state = _remember_print_seconds(state, base_seconds)
    state["db_exposing"] = True
    state["db_mode"] = mode
    state["db_seconds_left"] = float(seconds)
    state["db_total_seconds"] = seconds
    state["db_tick_seconds"] = TICK_SECONDS
    state["db_feather_px"] = 4.0
    state["db_stamp"] = stamp
    state["db_shape"] = str(shape_id)
    state["db_tool_scale"] = tool_scale
    tint = (120, 200, 255) if mode == "dodge" else (255, 200, 90)
    # Compact cursor image; full-resolution stamp stays in state for exposure.
    cursor = stamp
    mx = max(int(stamp.shape[0]), int(stamp.shape[1]), 1)
    if mx > 240:
        cursor_scale = 240.0 / mx
        nh = max(1, int(round(stamp.shape[0] * cursor_scale)))
        nw = max(1, int(round(stamp.shape[1] * cursor_scale)))
        from digital_negative.dodge_burn import _resize_mask

        cursor = _resize_mask(stamp, nh, nw)
    state["db_stamp_url"] = stamp_to_png_data_url(cursor, tint=tint)
    state["db_stamp_frac"] = float(stamp.shape[1]) / float(max(w, 1))
    # Seed last pointer (and scroll size) so a resting card still applies.
    state["db_last_pos"] = [float(seed_nx), float(seed_ny)]
    state["db_applied_ticks"] = 0
    # Snapshot the print so we can describe darken/lighten when the timer ends.
    before = state.get("live_rgb")
    if before is not None:
        state["db_before_rgb"] = np.asarray(before).copy()
    ensure_accum(state, h, w)

    verb = "Dodging" if mode == "dodge" else "Burning"
    total_stops = relative_pass_stops(seconds, base_seconds, mode)
    status = (
        f"**{verb}** — {seconds}s of {base_seconds:g}s base · ~{total_stops:+.2f} stops if held still.  \n"
        f"_Wave over the highlighted **live print** (right). "
        f"**Scroll** to resize the card. "
        f"{'Adds enlarger light → darker.' if mode == 'burn' else 'Holds back light → lighter.'} "
        f"Result appears when the timer ends._"
    )
    timer_md = (
        f"**{verb}… {seconds}s** — look right → wave the card over the highlighted print\n\n"
        f"{_pass_math_md(base_seconds, seconds, mode)}"
    )
    if state.get("dn") is not None:
        summary = f"{_stage_banner('print', _locks(state))}\n\n{status}\n\n{_history_md(state['dn'])}"
    else:
        summary = status
    st, hi = _split_summary(summary)
    state = {**state, "summary_cache": summary, "viewer_mode": "live"}
    seed_pos = f"{seed_nx:.4f},{seed_ny:.4f},{tool_scale:.3f}"
    return (
        gr.update(label=LIVE_WAVE_LABEL),
        timer_md,
        st,
        hi,
        state,
        gr.update(active=True),
        _db_flag_html(state),
        seed_pos,
        _wave_banner_html(state),
    )


def _db_refresh_print(paper_id, print_exposure, print_grade, print_contrast, state, *, status_md: str):
    """Re-print with current accum; returns live, timer_md, status, history, state."""
    if state.get("development_full") is None:
        st, hi = _split_summary(status_md)
        return state.get("live_rgb"), status_md, st, hi, state

    t = state.get("transmittance_proxy")
    if t is None:
        full = state["development_full"].transmittance
        step = max(1, int(np.ceil(max(full.shape) / LIVE_MAX_SIDE)))
        t = np.ascontiguousarray(full[::step, ::step])
        state = {**state, "transmittance_proxy": t}

    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    result = print_negative(
        t,
        state["dn"],
        paper,
        base_exposure_seconds=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        local_stops=local_stops_from_state(state),
        commit=False,
        **_technique_from_state(state),
    )
    live_rgb = _to_rgb_u8(result.preview)
    strokes = state.get("db_strokes") or []
    left = float(state.get("db_seconds_left", 0.0))
    if state.get("db_exposing"):
        verb = "Dodging" if state.get("db_mode") == "dodge" else "Burning"
        secs = max(0, int(np.ceil(left)))
        timer_line = f"**{verb}… {secs}s** — keep waving the card over the live print"
    elif strokes:
        timer_line = f"**Ready** — {len(strokes)} local pass(es). Reset clears them."
    else:
        timer_line = (
            "**Ready** — cut a card, then **Start — wave over print** "
            "(highlighted on the right)."
        )

    summary = (
        f"{_stage_banner('print', _locks(state))}\n\n"
        f"{status_md}\n\n"
        f"**Print preview** {live_rgb.shape[1]}×{live_rgb.shape[0]}  \n"
        f"{paper.name} · g{float(print_grade):.1f} · {_print_timer_label(print_exposure)} · "
        f"local passes={len(strokes)}\n\n{_history_md(state['dn'])}"
    )
    state = {
        **state,
        "print_draft": result,
        "live_rgb": live_rgb,
        "summary_cache": summary,
        "viewer_mode": "live",
        "strip_slots": list(STRIP_DEFAULT_SLOTS),
    }
    state = _remember_print_seconds(state, print_exposure)
    st, hi = _split_summary(summary)
    return live_rgb, timer_line, st, hi, state


def tick_dodge_burn(paper_id, print_exposure, print_grade, print_contrast, pos, state):
    if not state or not state.get("db_exposing"):
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            state,
            gr.update(active=False),
            _db_flag_html(state),
            _wave_banner_html(state),
        )

    h, w = _db_target_shape(state)
    parsed = parse_pointer_state(pos)
    pointer = None if parsed is None else (parsed[0], parsed[1])
    if parsed is not None:
        state = {**state, "db_tool_scale": float(parsed[2])}
    apply_exposure_tick(state, None, height=h, width=w, position=pointer)
    left = float(state.get("db_seconds_left", 0.0))
    still = bool(state.get("db_exposing"))
    secs = max(0, int(np.ceil(left)))
    applied = int(state.get("db_applied_ticks", 0))

    if still:
        verb = "Dodging" if state.get("db_mode") == "dodge" else "Burning"
        contact = "card on print" if applied else "move onto the print"
        timer_md = f"**{verb}… {secs}s** — {contact}"
        return (
            gr.update(label=LIVE_WAVE_LABEL),
            timer_md,
            gr.skip(),
            gr.skip(),
            state,
            gr.update(active=True),
            _db_flag_html(state),
            _wave_banner_html(state),
        )

    ls = local_stops_from_state(state)
    peak = float(np.max(np.abs(ls))) if ls is not None else 0.0
    if peak < 1e-4:
        status = (
            "**No local exposure recorded** — the card never registered over the print. "
            "Start again and keep the pointer on the highlighted print while it counts."
        )
        # Drop empty stroke noise
        strokes = state.get("db_strokes") or []
        if strokes and int(strokes[-1].get("applied_ticks", 0) or 0) == 0:
            state["db_strokes"] = strokes[:-1]
    else:
        mode = state.get("db_mode", "burn")
        verb = "Dodge" if mode == "dodge" else "Burn"
        expect = "lighter" if mode == "dodge" else "darker"
        status = (
            f"**Exposure done** — {verb} peak **{peak:.2f} stops** "
            f"(should read **{expect}** on the print). "
            "Start again for another pass, or Reset."
        )

    live, timer_md, st, hi, state = _db_refresh_print(
        paper_id, print_exposure, print_grade, print_contrast, state, status_md=status
    )
    before = state.pop("db_before_rgb", None)
    if before is not None and live is not None and peak >= 1e-4:
        try:
            b = float(np.mean(np.asarray(before, dtype=np.float32)))
            a = float(np.mean(np.asarray(live, dtype=np.float32)))
            if b > 1.5:
                b, a = b / 255.0, a / 255.0
            delta = a - b
            if abs(delta) >= 0.002:
                word = "lightened" if delta > 0 else "darkened"
                pct = abs(delta) / max(b, 1e-6) * 100.0
                note = f" Print mean **{word} ~{pct:.1f}%** vs pre-pass."
                st = f"{st}\n\n{note}" if isinstance(st, str) else st
                if isinstance(state.get("summary_cache"), str):
                    state["summary_cache"] = state["summary_cache"] + f"\n\n{note}"
        except Exception:
            pass
    return (
        gr.update(value=live, label=LIVE_PRINT_LABEL),
        timer_md,
        st,
        hi,
        state,
        gr.update(active=False),
        _db_flag_html(state),
        _wave_banner_html(state),
    )


def reset_dodge_burn(paper_id, print_exposure, print_grade, print_contrast, state):
    if not state:
        raise gr.Error("Commit Ingest / Develop first.")
    if state.get("db_exposing"):
        raise gr.Error("Stop — wait for the current exposure, or unlock and reset.")
    state = reset_local_work({**state})
    if state.get("dn") is not None:
        state["dn"].metadata.setdefault("print", {})["dodge_burn"] = []
    live_rgb, timer_line, status, history, state = _db_refresh_print(
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        status_md="**Local work cleared** — base print only (global exposure / grade kept).",
    )
    editor = _editor_from_print(live_rgb)
    return (
        gr.update(value=live_rgb, label=LIVE_PRINT_LABEL),
        timer_line,
        status,
        history,
        editor,
        state,
        gr.update(active=False),
        _db_flag_html(state),
        "",
        _wave_banner_html(state),
    )


def unlock_develop(state):
    if not state or state.get("dn") is None:
        raise gr.Error("Nothing to unlock — Commit Ingest first.")
    if not _locked(state, "development"):
        raise gr.Error("Develop is not locked.")

    dn = state["dn"]
    if _locked(state, "print"):
        _unlock_stage(dn, "print")
    _unlock_stage(dn, "development")

    state = {
        **state,
        "development_full": None,
        "transmittance_proxy": None,
        "print": None,
        "print_draft": None,
        "stage": "development",
    }
    state = reset_local_work(state)
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Develop unlocked** — adjust Develop below, then Commit again.\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        state.get("original_ref"),
        state.get("latent_ref"),
        state.get("neg_ref"),
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        on,
        off,
        off,
        on,
        on,
        on,
        on,
        off,
        gr.update(open=True),
        gr.update(open=True),
        # download affordances no longer match the working state
        gr.update(visible=False),
        "",
        state,
        "develop",
    )

def seed_dodge_burn_editor(state):
    """After Commit Develop — prepare the enlarger-card controls."""
    rgb = state.get("live_rgb") if state else None
    editor = _editor_from_print(rgb)
    base = float((state or {}).get("print_base_seconds") or REFERENCE_BASE_SECONDS)
    return (
        gr.update(value=editor, visible=False),
        "**Ready** — pick a card shape (oval is fine), set pass seconds, then "
        "**Start — wave over print →**. The print on the right is the easel.",
        _base_math_md(base),
        _pass_math_md(base, 4.0, "burn"),
        "soft_oval",
    )


def guided_first_print(
    sample_path,
    file_obj,
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    exposure_index,
    contrast_filter,
    scene_exposure,
    halation,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
):
    """One-click path: Ingest → Develop → Print easel ready with a soft-oval card."""
    film_u = gr.skip()
    developer_u = gr.skip()
    minutes_u = gr.skip()
    contrast_u = gr.skip()
    grain_u = gr.skip()
    unlock_dev_u = gr.skip()
    unlock_print_u = gr.skip()

    if state is None or state.get("dn") is None or not _locked(state, "ingest"):
        (
            live_rgb,
            original_ref,
            latent_ref,
            neg_ref,
            status,
            history,
            sample_u,
            file_u,
            ingest_btn_u,
            develop_btn_u,
            print_btn_u,
            ingest_acc_u,
            develop_acc_u,
            print_acc_u,
            rot_ccw,
            rot_180,
            rot_cw,
            apply_frame_u,
            reset_frame_u,
            auto_crop_u,
            auto_straighten_u,
            straighten_u,
            crop_rect_u,
            crop_ratio_u,
            inspect_out,
            state,
        ) = commit_ingest(sample_path, file_obj, state)
    else:
        live_rgb = state.get("live_rgb")
        original_ref = state.get("original_ref")
        latent_ref = state.get("latent_ref")
        neg_ref = state.get("neg_ref")
        status, history = _split_summary(state.get("summary_cache", ""))
        sample_u = gr.skip()
        file_u = gr.skip()
        ingest_btn_u = gr.update(interactive=False)
        develop_btn_u = gr.update(interactive=True)
        print_btn_u = gr.update(interactive=False)
        ingest_acc_u = gr.update(open=False)
        develop_acc_u = gr.update(open=True)
        print_acc_u = gr.update(open=True)
        rot_ccw = gr.update(interactive=True)
        rot_180 = gr.update(interactive=True)
        rot_cw = gr.update(interactive=True)
        apply_frame_u = gr.update(interactive=True)
        reset_frame_u = gr.update(interactive=True)
        auto_crop_u = gr.update(interactive=True)
        auto_straighten_u = gr.update(interactive=True)
        straighten_u = gr.skip()
        crop_rect_u = gr.skip()
        crop_ratio_u = gr.skip()
        inspect_out = state.get("live_inspect")
        if inspect_out is None:
            inspect_out = live_rgb
        # inspect_acc removed — Inspect is a live-preview tool mode

    if not _locked(state, "development"):
        (
            live_rgb,
            original_ref,
            latent_ref,
            neg_ref,
            status,
            history,
            film_u,
            developer_u,
            minutes_u,
            contrast_u,
            grain_u,
            develop_btn_u,
            unlock_dev_u,
            print_btn_u,
            unlock_print_u,
            develop_acc_u,
            print_acc_u,
            state,
        ) = commit_develop(
            film_id,
            developer_id,
            development_minutes,
            contrast,
            grain,
            exposure_index,
            contrast_filter,
            scene_exposure,
            halation,
            state,
        )
    else:
        develop_btn_u = gr.update(interactive=False)
        unlock_dev_u = gr.update(interactive=True)
        print_btn_u = gr.update(interactive=True)
        unlock_print_u = gr.update(interactive=_locked(state, "print"))
        develop_acc_u = gr.update(open=False)
        print_acc_u = gr.update(open=True)

    packed = live_preview_high(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        exposure_index,
        contrast_filter,
        scene_exposure,
        halation,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        False, 0.0, 5.0, 4.5, 3.5,
        False, 5, 0.5, 0.0, 0.0, "none", 0.0,
        state,
    )
    live_rgb, original_ref, latent_ref, neg_ref, status, history, inspect_out, state = packed
    state = _remember_print_seconds(state, print_exposure)
    if not state.get("db_exposing"):
        state = reset_local_work({**state})

    guide = (
        f"{_stage_banner('print', _locks(state))}\n\n"
        f"**First-print guide ready — you’re on the easel**  \n"
        f"1. Base timer **{float(print_exposure):g}s** (change if you want)  \n"
        f"2. Soft-oval card loaded — **scroll** over the print to resize  \n"
        f"3. Mode defaults to **burn (darker)** — switch to dodge for lighter  \n"
        f"4. **Start — wave over print →** and move on the highlighted print  \n"
        f"5. When the timer ends, read the before/after note · Reset clears the pass\n\n"
        f"{_history_md(state['dn'])}"
    )
    state["summary_cache"] = guide
    st, hi = _split_summary(guide)
    timer_md = (
        "**Ready for the easel** — soft oval loaded. Scroll to size, then Start.\n\n"
        f"{_pass_math_md(print_exposure, 4.0, 'burn')}"
    )
    return (
        live_rgb,
        original_ref,
        latent_ref,
        neg_ref,
        st,
        hi,
        sample_u,
        file_u,
        ingest_btn_u,
        film_u,
        developer_u,
        minutes_u,
        contrast_u,
        grain_u,
        develop_btn_u,
        unlock_dev_u,
        print_btn_u,
        unlock_print_u,
        ingest_acc_u,
        develop_acc_u,
        print_acc_u,
        rot_ccw,
        rot_180,
        rot_cw,
        apply_frame_u,
        reset_frame_u,
        auto_crop_u,
        auto_straighten_u,
        straighten_u,
        crop_rect_u,
        crop_ratio_u,
        inspect_out,
        state,
        gr.update(value=_editor_from_print(live_rgb), visible=False),
        timer_md,
        _base_math_md(print_exposure),
        _pass_math_md(print_exposure, 4.0, "burn"),
        "soft_oval",
        _wave_banner_html(state),
    )


def unlock_print(state):
    if not state or state.get("dn") is None:
        raise gr.Error("Nothing to unlock.")
    if not _locked(state, "print"):
        raise gr.Error("Print is not locked.")
    if not _locked(state, "development"):
        raise gr.Error("Develop must stay committed to revise Print.")

    dn = state["dn"]
    _unlock_stage(dn, "print")
    state = {**state, "print": None, "stage": "print"}
    summary = (
        f"{_stage_banner('print', _locks(state))}\n\n"
        f"**Print unlocked** — adjust Print below, then Commit Print.\n\n"
        f"{_history_md(dn)}"
    )
    state["summary_cache"] = summary
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        state.get("live_rgb"),
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        off,
        gr.update(open=True),
        gr.update(visible=False),
        "",
        state,
    )


def reset_session():
    summary = (
        "**1. Upload — working** → 2. Develop → 3. Print\n\n"
        "*Add photos to the camera roll to begin.*"
    )
    on, off = gr.update(interactive=True), gr.update(interactive=False)
    return (
        None,
        None,
        None,
        None,
        *_split_summary(summary),
        on,
        on,
        on,
        on,
        on,
        on,
        on,
        on,
        off,
        off,
        on,
        on,
        on,
        on,
        off,
        off,
        gr.update(open=True),
        gr.update(open=True),
        gr.update(open=True),
        off,
        off,
        off,
        off,
        off,
        off,
        off,
        0.0,
        DEFAULT_CROP_RECT,
        "free",
        gr.update(visible=False),
        "",
        {"roll": [], "roll_index": -1},
        "ingest",
        _roll_meta_md({"roll": [], "roll_index": -1}),
        _roll_gallery_update({"roll": [], "roll_index": -1}),
        off,
    )


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        active_drawer = gr.Textbox(value="ingest", elem_id="active_drawer", show_label=False)
        gr.Markdown("# Digital Negative Darkroom", elem_id="app_header")

        with gr.Row(elem_id="main_workspace", equal_height=False):
            # ——— Icon rail ———
            with gr.Column(scale=0, elem_id="icon_rail", min_width=56):
                rail_ingest = gr.Button(
                    "⬇\nUpload", elem_id="rail_ingest", size="sm",
                    elem_classes=["rail-btn", "rail-active"],
                )
                rail_roll = gr.Button(
                    "▤\nRoll", elem_id="rail_roll", size="sm", elem_classes=["rail-btn"]
                )
                rail_develop = gr.Button(
                    "⚗\nDev", elem_id="rail_develop", size="sm", elem_classes=["rail-btn"]
                )
                rail_print = gr.Button(
                    "▣\nPrint", elem_id="rail_print", size="sm", elem_classes=["rail-btn"]
                )
                rail_frame = gr.Button(
                    "✂\nFrame", elem_id="rail_frame", size="sm", elem_classes=["rail-btn"]
                )
                rail_log = gr.Button(
                    "☰\nLog", elem_id="rail_log", size="sm", elem_classes=["rail-btn"]
                )
                gr.HTML('<div class="rail-spacer"></div>')
                reset_btn = gr.Button("+\nNew", elem_id="rail_new", size="sm")

            # ——— Drawer panels (one visible) ———
            with gr.Column(scale=0, elem_id="drawer_host", min_width=0):
                with gr.Group(elem_id="drawer_ingest", elem_classes=["drawer-panel", "is-open"]):
                    with gr.Accordion("Upload", open=True, elem_id="acc_ingest") as ingest_acc:
                        sample = gr.Dropdown(
                            choices=SAMPLE_CHOICES,
                            value=default_sample,
                            label="Sample",
                        )
                        file_in = gr.File(
                            label="Photos",
                            file_count="multiple",
                            file_types=[
                                ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2",
                                ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp",
                                ".heic", ".heif", ".avif",
                            ],
                            height=140,
                            elem_id="ingest_upload",
                        )
                        ingest_btn = gr.Button("Add to roll", variant="primary", size="sm")
                        gr.Markdown(
                            "_Photos land in the **Roll** tab._",
                            elem_id="ingest_hint",
                        )

                with gr.Group(elem_id="drawer_roll", elem_classes=["drawer-panel"]):
                    with gr.Accordion("Camera roll", open=True, elem_id="acc_roll") as roll_acc:
                        roll_meta = gr.Markdown(
                            _roll_meta_md(None),
                            elem_id="roll_meta",
                        )
                        roll_gallery = gr.HTML(
                            value=_roll_gallery_update(None),
                            elem_id="camera_roll",
                        )
                        # Off-screen triggers (CSS-parked, not visible=False) so JS
                        # can write values Gradio will actually submit.
                        roll_remove_index = gr.Textbox(
                            value="",
                            show_label=False,
                            interactive=True,
                            elem_id="roll_remove_index",
                        )
                        remove_roll_btn = gr.Button(
                            "Remove",
                            interactive=True,
                            elem_id="roll_remove",
                        )
                        roll_switch_index = gr.Textbox(
                            value="-1",
                            show_label=False,
                            interactive=True,
                            elem_id="roll_switch_index",
                        )
                        roll_switch_btn = gr.Button(
                            "Switch",
                            interactive=True,
                            elem_id="roll_switch",
                        )
                        roll_pending_index = gr.Number(
                            value=-1,
                            show_label=False,
                            interactive=True,
                            elem_id="roll_pending_index",
                        )

                with gr.Group(elem_id="drawer_develop", elem_classes=["drawer-panel"]):
                    with gr.Accordion("Develop", open=True, elem_id="acc_develop") as develop_acc:
                        chemistry_mode = gr.Radio(
                            choices=CHEMISTRY_MODE_LABELS,
                            value="bw",
                            label="Chemistry",
                            elem_id="chemistry_mode",
                        )
                        film = gr.Dropdown(
                            choices=FILM_CHOICES_BW,
                            value=FILM_CHOICES_BW[0][1] if FILM_CHOICES_BW else None,
                            label="Film",
                        )
                        developer = gr.Dropdown(
                            choices=_INIT_DEV_CHOICES,
                            value=_INIT_DEV_ID,
                            label="Developer",
                        )
                        development_minutes = gr.Slider(
                            _DEV_TIME_SLIDER_MIN,
                            _DEV_TIME_SLIDER_MAX,
                            value=_INIT_TNORM,
                            step=0.25,
                            label=f"Dev time · N={_INIT_TNORM:g}",
                        )
                        contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast")
                        grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain")
                        exposure_index = gr.Slider(
                            25, 6400, value=400, step=25, label="Exposure index (EI)"
                        )
                        contrast_filter = gr.Dropdown(
                            choices=FILTER_LABELS,
                            value="none",
                            label="Contrast filter",
                        )
                        scene_exposure = gr.Slider(
                            0.01, 60.0, value=0.01, step=0.01,
                            label="Scene shutter (s)",
                        )
                        halation = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Halation")
                        with gr.Row(elem_id="develop_commit_row"):
                            develop_btn = gr.Button(
                                "Commit Develop", interactive=False, variant="primary", size="sm"
                            )
                            unlock_develop_btn = gr.Button("Unlock", interactive=False, size="sm")

                with gr.Group(elem_id="drawer_print", elem_classes=["drawer-panel"]):
                    with gr.Accordion("Print", open=True, elem_id="acc_print") as print_acc:
                        paper = gr.Dropdown(
                            choices=PAPER_CHOICES_BW,
                            value=PAPER_CHOICES_BW[0][1] if PAPER_CHOICES_BW else None,
                            label="Paper",
                        )
                        print_exposure = gr.Slider(
                            2.0, 64.0, value=8.0, step=0.5, label="Base exposure (s)"
                        )
                        base_math_md = gr.Markdown(_base_math_md(8.0), elem_id="base_math")
                        print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="MG grade")
                        print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Filter")
                        with gr.Row(elem_id="cc_row"):
                            cc_cyan = gr.Slider(0, 100, value=0, step=1, label="CC Cyan")
                            cc_magenta = gr.Slider(0, 100, value=0, step=1, label="CC Magenta")
                            cc_yellow = gr.Slider(0, 100, value=0, step=1, label="CC Yellow")
                        split_grade = gr.Checkbox(label="Split-grade", value=False)
                        with gr.Row():
                            soft_grade = gr.Slider(0.0, 5.0, value=0.0, step=0.5, label="Soft grade")
                            hard_grade = gr.Slider(0.0, 5.0, value=5.0, step=0.5, label="Hard grade")
                        with gr.Row():
                            soft_seconds = gr.Slider(1.0, 64.0, value=4.5, step=0.5, label="Soft (s)")
                            hard_seconds = gr.Slider(1.0, 64.0, value=3.5, step=0.5, label="Hard (s)")
                        test_strips = gr.Checkbox(label="Test strips", value=False)
                        with gr.Row():
                            test_bands = gr.Slider(3, 9, value=5, step=1, label="Bands")
                            test_stops = gr.Slider(0.25, 1.0, value=0.5, step=0.25, label="Band stops")
                        flash_stops = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Flash (stops)")
                        dry_down = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Dry-down %")
                        tone = gr.Dropdown(choices=TONE_LABELS, value="none", label="Tone")
                        border_frac = gr.Slider(0.0, 0.12, value=0.0, step=0.005, label="Border")
                        with gr.Row(elem_id="print_commit_row"):
                            print_btn = gr.Button(
                                "Commit Print", interactive=False, variant="primary", size="sm"
                            )
                            unlock_print_btn = gr.Button("Unlock", interactive=False, size="sm")
                        # One visible trigger. With a single package it
                        # downloads straight away; with several it opens a
                        # popup listing them. The real DownloadButtons sit
                        # off-screen and are clicked by the menu.
                        download_trigger = gr.Button(
                            "⇣ Download",
                            visible=False,
                            size="sm",
                            elem_id="download_trigger",
                        )
                        download_modes = gr.Textbox(
                            value="", elem_id="download_modes", show_label=False
                        )
                        dl_pkg_print = gr.DownloadButton(
                            "print", size="sm", elem_id="dl_pkg_print"
                        )
                        dl_pkg_both = gr.DownloadButton(
                            "both", size="sm", elem_id="dl_pkg_both"
                        )
                        dl_pkg_negative = gr.DownloadButton(
                            "negative", size="sm", elem_id="dl_pkg_negative"
                        )
                        gr.Markdown(
                            "_Dodge / burn: **right-click the print** → Dodge or Burn._",
                            elem_id="db_hint",
                        )

                with gr.Group(elem_id="drawer_frame", elem_classes=["drawer-panel"]):
                    with gr.Accordion("Frame", open=True, elem_id="acc_frame") as frame_acc:
                        gr.Markdown(
                            "_Framing lives in **Modules → Crop & straighten** "
                            "(also **right-click the print**). "
                            "Rotate 90°, auto-straighten, crop, then Apply._",
                            elem_id="crop_hint",
                        )

                with gr.Group(elem_id="drawer_log", elem_classes=["drawer-panel"]):
                    with gr.Accordion("Decision log", open=True, elem_id="acc_log") as log_acc:
                        history = gr.Markdown(
                            "_Locked decisions only — exploring does not write here._",
                            elem_id="history_box",
                        )

            # ——— Live theoretical print (fills remaining viewport) ———
            with gr.Column(scale=1, elem_id="preview_col", min_width=420):
                db_wave_banner = gr.HTML(_wave_banner_html(None), elem_id="db_wave_banner")
                db_size_readout = gr.HTML(
                    '<div class="db-size-pill"><span class="db-tool-mode">Print</span> · '
                    'card <strong class="db-size-value">100%</strong> · right-click for tools</div>',
                    elem_id="db_size_readout",
                )
                # Stage + recipe readout floats over the print rather than
                # taking a slice of the drawer above the controls.
                status = gr.Markdown(
                    "**1. Upload — working** → 2. Develop → 3. Print  \n"
                    "_Add photos from **Upload** to begin._",
                    elem_id="ritual_status",
                )
                spot_readout = gr.Markdown(
                    "_Hover the print for Zone / density._",
                    elem_id="spot_readout",
                )
                spot_pos = gr.Textbox(value="0.5000,0.5000", elem_id="spot_pos", show_label=False)
                preview_tool = gr.Radio(
                    choices=[
                        ("print", "print"),
                        ("frame", "frame"),
                        ("inspect", "inspect"),
                    ],
                    value="print",
                    label="preview_tool",
                    elem_id="preview_tool",
                    show_label=False,
                )
                live_out = gr.Image(
                    label=LIVE_PRINT_LABEL + " · easel",
                    type="numpy",
                    elem_id="live_preview",
                    height=720,
                    buttons=[],
                )

                # Filmstrip: three slots holding whatever isn't in the preview.
                # Clicking one swaps it with the preview; the label shows on hover.
                with gr.Row(elem_id="seq_strip"):
                    original_out = gr.Image(
                        label="Original", type="numpy", height=44, buttons=[],
                        elem_classes=["seq-thumb"],
                    )
                    latent_out = gr.Image(
                        label="Latent DN", type="numpy", height=44, buttons=[],
                        elem_classes=["seq-thumb"],
                    )
                    neg_out = gr.Image(
                        label="Negative", type="numpy", height=44, buttons=[],
                        elem_classes=["seq-thumb"],
                    )

                inspect_out = gr.Image(
                    label="Inspect",
                    type="numpy",
                    elem_id="inspect_preview",
                    visible=False,
                    buttons=["fullscreen", "download"],
                )

            # ——— Persistent module panel (darktable-style, right side) ———
            with gr.Column(scale=0, elem_id="module_panel", min_width=280):
                gr.HTML('<div class="module_panel_title">Modules</div>')
                with gr.Accordion("Inspect · zoom", open=False, elem_id="mod_inspect"):
                    gr.Markdown(
                        "Scroll to zoom · drag to pan when zoomed · double-click resets. "
                        "Hover the print for a Zone / density spot reading."
                    )
                    hist_plot = gr.Image(
                        label="Histogram",
                        type="numpy",
                        elem_id="hist_plot",
                        height=140,
                        buttons=[],
                        show_label=False,
                    )
                    inspect_tip = gr.Markdown(
                        "_Histogram appears once a print preview exists._",
                        elem_id="inspect_tip",
                    )
                    with gr.Row():
                        clip_hi = gr.Checkbox(label="Blown (Z VII+)", value=False)
                        clip_lo = gr.Checkbox(label="Crushed (Z I−)", value=False)
                        fit_tones_btn = gr.Button(
                            "Fit to paper",
                            size="sm",
                            variant="secondary",
                            elem_id="fit_tones_btn",
                        )
                    with gr.Row():
                        pin_ab_btn = gr.Button("Pin A", size="sm")
                        toggle_ab_btn = gr.Button("Show A", size="sm", interactive=False)
                    inspect_open = gr.Textbox(value="", elem_id="inspect_open", show_label=False)

                with gr.Accordion("Curves", open=False, elem_id="mod_curves"):
                    curve_summary = gr.Markdown(
                        "_Commit Ingest, then refresh to see where this frame "
                        "lands on the film and paper curves._",
                        elem_id="curve_summary",
                    )
                    curve_plot = gr.Image(
                        label="Curves",
                        type="numpy",
                        elem_id="curve_plot",
                        height=300,
                        buttons=["fullscreen"],
                        show_label=False,
                    )
                    curve_refresh_btn = gr.Button(
                        "Refresh curves", size="sm", elem_id="curve_refresh"
                    )
                    # Bumped by JS when the module expands, so the plot is
                    # rebuilt from current settings on open.
                    curves_open = gr.Textbox(
                        value="", elem_id="curves_open", show_label=False
                    )

                with gr.Accordion("Recipes", open=False, elem_id="mod_recipes"):
                    recipe_name = gr.Textbox(
                        value="session", label="Name", elem_id="recipe_name_box"
                    )
                    with gr.Row():
                        save_recipe_btn = gr.Button("Save recipe", size="sm")
                        recipe_download = gr.DownloadButton(
                            "recipe.json", size="sm", elem_id="recipe_download"
                        )
                    recipe_file = gr.File(
                        label="Load recipe",
                        file_types=[".json"],
                        height=80,
                        elem_id="recipe_upload",
                    )
                    load_recipe_btn = gr.Button("Apply recipe", size="sm")
                    recipe_tip = gr.Markdown(
                        "_Save film / chemistry / print controls as JSON._"
                    )

                with gr.Accordion("Dodge & burn", open=False, elem_id="mod_dodge_burn"):
                    db_shape = gr.Radio(
                        choices=[(label, key) for key, label in CARD_PRESETS],
                        value="soft_oval",
                        label="Card shape",
                    )
                    db_editor = gr.ImageEditor(
                        label="Custom card (paint only if shape = Custom)",
                        type="numpy",
                        image_mode="RGBA",
                        height=160,
                        value=tool_workshop_canvas(),
                        brush=gr.Brush(
                            default_size=48,
                            colors=["#e0954f", "#ffffff", "#6fd1c7"],
                            default_color="#e0954f",
                            color_mode="defaults",
                        ),
                        eraser=gr.Eraser(),
                        layers=True,
                        transforms=(),
                        sources=(),
                        buttons=["fullscreen"],
                        visible=False,
                    )
                    db_mode = gr.Radio(
                        choices=[
                            ("Dodge — hold back light (lighter)", "dodge"),
                            ("Burn — add enlarger light (darker)", "burn"),
                        ],
                        value="burn",
                        label="Mode",
                    )
                    db_seconds = gr.Slider(
                        1, 32, value=4, step=1, label="Pass (s)"
                    )
                    pass_math_md = gr.Markdown(_pass_math_md(8.0, 4.0, "burn"), elem_id="pass_math")
                    db_timer_md = gr.Markdown("**Ready** — Start, then wave over the print")
                    with gr.Row(elem_id="db_actions"):
                        db_start_btn = gr.Button(
                            "Start — wave over print →", variant="primary", size="sm"
                        )
                        db_reset_btn = gr.Button("Reset local work", size="sm")
                    db_flag = gr.HTML(_db_flag_html(None), elem_id="db_flag")
                    db_pos = gr.Textbox(value="0.5000,0.5000", elem_id="db_pos", show_label=False)
                    with gr.Column(elem_classes=["db_clock_hidden"]):
                        db_clock = gr.Timer(value=TICK_SECONDS, active=False)

                with gr.Accordion("Crop & straighten", open=False, elem_id="mod_crop"):
                    crop_hint = gr.Markdown(
                        "_Rotate if needed, auto-straighten / crop, then Apply._",
                        elem_id="crop_float_hint",
                    )
                    with gr.Row(elem_id="rotate_row"):
                        rotate_ccw_btn = gr.Button(
                            "⟲ 90°", size="sm", interactive=False, elem_id="rotate_ccw_btn"
                        )
                        rotate_180_btn = gr.Button(
                            "180°", size="sm", interactive=False, elem_id="rotate_180_btn"
                        )
                        rotate_cw_btn = gr.Button(
                            "90° ⟳", size="sm", interactive=False, elem_id="rotate_cw_btn"
                        )
                    crop_ratio = gr.Radio(
                        choices=CROP_RATIO_CHOICES,
                        value="free",
                        label="Aspect ratio",
                        elem_id="crop_ratio",
                    )
                    with gr.Row():
                        straighten_deg = gr.Slider(
                            -15.0, 15.0, value=0.0, step=0.1, label="Straighten °", scale=3
                        )
                        auto_straighten_btn = gr.Button(
                            "Auto",
                            interactive=False,
                            variant="secondary",
                            size="sm",
                            scale=1,
                            elem_id="auto_straighten_btn",
                        )
                    with gr.Row():
                        auto_crop_rule = gr.Dropdown(
                            choices=AUTO_CROP_RULE_CHOICES,
                            value="auto",
                            label="Rule",
                            scale=3,
                        )
                        auto_crop_btn = gr.Button(
                            "Auto crop",
                            interactive=False,
                            variant="secondary",
                            size="sm",
                            scale=1,
                            elem_id="auto_crop_btn",
                        )
                    crop_rect = gr.Textbox(
                        value=DEFAULT_CROP_RECT,
                        label="crop_rect",
                        elem_id="crop_rect",
                        show_label=False,
                    )
                    with gr.Row():
                        apply_framing_btn = gr.Button(
                            "Apply framing", interactive=False, variant="secondary", size="sm"
                        )
                        reset_framing_btn = gr.Button("Reset framing", interactive=False, size="sm")

        # Compatibility aliases used by older handlers expecting frame_tools group
        frame_tools = frame_acc

        with gr.Group(visible=False, elem_id="roll_save_modal") as roll_save_modal:
            with gr.Column(elem_id="roll_save_dialog"):
                gr.Markdown(
                    "**Save changes to this frame?**\n\n"
                    "Develop and print adjustments on the current photo can be "
                    "kept before you switch, or discarded."
                )
                with gr.Row(elem_id="roll_save_actions"):
                    roll_save_btn = gr.Button("Save & switch", variant="primary", size="sm")
                    roll_discard_btn = gr.Button("Discard", size="sm")
                    roll_cancel_btn = gr.Button("Cancel", size="sm")

        # Always pass develop + print controls so the large viewer can show a
        # theoretical print through the working negative while developing.
        preview_inputs = [
            chemistry_mode,
            film,
            developer,
            development_minutes,
            contrast,
            grain,
            exposure_index,
            contrast_filter,
            scene_exposure,
            halation,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            split_grade,
            soft_grade,
            hard_grade,
            soft_seconds,
            hard_seconds,
            test_strips,
            test_bands,
            test_stops,
            flash_stops,
            dry_down,
            tone,
            border_frac,
            cc_cyan,
            cc_magenta,
            cc_yellow,
            state,
        ]
        preview_outputs = [
            live_out, original_out, latent_out, neg_out, status, history, inspect_out, state
        ]

        ingest_outputs = [
            live_out, original_out, latent_out, neg_out, status, history,
            sample, file_in, ingest_btn, develop_btn, print_btn,
            unlock_develop_btn, unlock_print_btn,
            ingest_acc, develop_acc, print_acc,
            rotate_ccw_btn, rotate_180_btn, rotate_cw_btn,
            apply_framing_btn, reset_framing_btn, auto_crop_btn, auto_straighten_btn,
            straighten_deg, crop_rect, crop_ratio,
            inspect_out, state, active_drawer,
            roll_meta, roll_gallery, remove_roll_btn,
        ]
        control_outputs = [
            chemistry_mode,
            film,
            developer,
            development_minutes,
            contrast,
            grain,
            exposure_index,
            contrast_filter,
            scene_exposure,
            halation,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            split_grade,
            soft_grade,
            hard_grade,
            soft_seconds,
            hard_seconds,
            test_strips,
            test_bands,
            test_stops,
            flash_stops,
            dry_down,
            tone,
            border_frac,
            cc_cyan,
            cc_magenta,
            cc_yellow,
        ]
        # Ingest/remove also restore Develop/Print interactivity — otherwise a
        # prior Commit Develop leaves film controls disabled on the new frame.
        session_control_outputs = ingest_outputs + control_outputs
        roll_switch_outputs = session_control_outputs + [
            roll_save_modal,
            roll_pending_index,
        ]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=session_control_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        # Uploading appends to the camera roll immediately (multi-file OK).
        file_in.upload(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=session_control_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        # Primary path: JS writes index:token → textbox .change (same as remove).
        # Button click is only a fallback; do not fire both (stale-index race).
        # Control inputs snapshot the outgoing frame so settings stay per-frame.
        roll_switch_index.change(
            fn=begin_roll_switch,
            inputs=[roll_switch_index, state] + control_outputs,
            outputs=roll_switch_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        roll_switch_btn.click(
            fn=begin_roll_switch,
            inputs=[roll_switch_index, state] + control_outputs,
            outputs=roll_switch_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        roll_save_btn.click(
            fn=save_and_switch_roll,
            inputs=[roll_pending_index, state] + control_outputs,
            outputs=roll_switch_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        roll_discard_btn.click(
            fn=discard_and_switch_roll,
            inputs=[roll_pending_index, state],
            outputs=roll_switch_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        roll_cancel_btn.click(
            fn=cancel_roll_switch,
            inputs=[state],
            outputs=roll_switch_outputs,
        )

        # Primary path: JS writes index:token into the textbox (always a new value).
        roll_remove_index.change(
            fn=remove_from_roll,
            inputs=[roll_remove_index, state],
            outputs=session_control_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        # Fallback if only the hidden button receives the click.
        remove_roll_btn.click(
            fn=remove_from_roll,
            inputs=[roll_remove_index, state],
            outputs=session_control_outputs,
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        chemistry_mode.change(
            fn=on_chemistry_mode_change,
            inputs=[chemistry_mode],
            outputs=[film, developer, development_minutes, exposure_index, paper],
        ).then(
            fn=live_preview_edit,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        for ctrl in (
            development_minutes,
            contrast,
            grain,
            exposure_index,
            contrast_filter,
            scene_exposure,
            halation,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            split_grade,
            soft_grade,
            hard_grade,
            soft_seconds,
            hard_seconds,
            test_strips,
            test_bands,
            test_stops,
            flash_stops,
            dry_down,
            tone,
            border_frac,
            cc_cyan,
            cc_magenta,
            cc_yellow,
        ):
            # Drag = fast lower-res; release/change = commit-quality preview
            ctrl.input(fn=live_preview_drag, inputs=preview_inputs, outputs=preview_outputs)
            ctrl.change(fn=live_preview_edit, inputs=preview_inputs, outputs=preview_outputs)

        def _sync_db_pass_timer(base_seconds, mode, current_pass):
            base = max(2.0, float(base_seconds))
            cur = int(round(float(current_pass)))
            if str(mode).lower().startswith("dodge"):
                mx = max(1, int(round(base)))
                value = min(cur, mx)
            else:
                mx = max(1, int(round(base * 2)))
                value = min(cur, mx)
            return (
                gr.update(maximum=mx, value=value),
                _base_math_md(base),
                _pass_math_md(base, value, mode),
            )

        def _sync_pass_math_only(base_seconds, mode, current_pass):
            return _pass_math_md(base_seconds, current_pass, mode)

        def _toggle_custom_editor(shape_id):
            return gr.update(visible=str(shape_id).lower() == "custom")

        def _on_shape_change(shape_id, mode, state):
            editor = _toggle_custom_editor(shape_id)
            state, flag = _sync_tool_preview(shape_id, mode, state)
            return editor, state, flag

        def _on_mode_change(base_seconds, mode, current_pass, shape_id, state):
            secs, base_md, pass_md = _sync_db_pass_timer(base_seconds, mode, current_pass)
            state, flag = _sync_tool_preview(shape_id, mode, state)
            return secs, base_md, pass_md, state, flag

        print_exposure.change(
            fn=_sync_db_pass_timer,
            inputs=[print_exposure, db_mode, db_seconds],
            outputs=[db_seconds, base_math_md, pass_math_md],
        )
        db_mode.change(
            fn=_on_mode_change,
            inputs=[print_exposure, db_mode, db_seconds, db_shape, state],
            outputs=[db_seconds, base_math_md, pass_math_md, state, db_flag],
        )
        db_seconds.change(
            fn=_sync_pass_math_only,
            inputs=[print_exposure, db_mode, db_seconds],
            outputs=[pass_math_md],
        )
        db_shape.change(
            fn=_on_shape_change,
            inputs=[db_shape, db_mode, state],
            outputs=[db_editor, state, db_flag],
        )

        # Film / developer swap chemistry list + datasheet-normal minutes, then refresh.
        film.change(
            fn=on_film_change,
            inputs=[film],
            outputs=[developer, development_minutes, exposure_index],
        ).then(
            fn=live_preview_edit,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        developer.change(
            fn=on_developer_change,
            inputs=[film, developer],
            outputs=[development_minutes],
        ).then(
            fn=live_preview_edit,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        develop_btn.click(
            fn=commit_develop,
            inputs=[
                film, developer, development_minutes, contrast, grain,
                exposure_index, contrast_filter, scene_exposure, halation, state,
            ],
            outputs=[
                download_trigger, download_modes, dl_pkg_negative,
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
                develop_acc, print_acc, state, active_drawer,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        ).then(
            fn=seed_dodge_burn_editor,
            inputs=[state],
            outputs=[db_editor, db_timer_md, base_math_md, pass_math_md, db_shape],
        )

        db_start_btn.click(
            fn=start_dodge_burn,
            inputs=[
                db_mode,
                db_seconds,
                db_shape,
                paper,
                print_exposure,
                print_grade,
                print_contrast,
                db_editor,
                db_pos,
                state,
            ],
            outputs=[
                live_out,
                db_timer_md,
                status,
                history,
                state,
                db_clock,
                db_flag,
                db_pos,
                db_wave_banner,
            ],
        )
        db_clock.tick(
            fn=tick_dodge_burn,
            inputs=[paper, print_exposure, print_grade, print_contrast, db_pos, state],
            outputs=[
                live_out,
                db_timer_md,
                status,
                history,
                state,
                db_clock,
                db_flag,
                db_wave_banner,
            ],
            js=DB_TICK_JS,
        )
        db_reset_btn.click(
            fn=reset_dodge_burn,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[
                live_out,
                db_timer_md,
                status,
                history,
                db_editor,
                state,
                db_clock,
                db_flag,
                db_pos,
                db_wave_banner,
            ],
        )


        unlock_develop_btn.click(
            fn=unlock_develop,
            inputs=[state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn,
                paper, print_exposure, print_grade, print_contrast, unlock_print_btn,
                develop_acc, print_acc,
                download_trigger, download_modes,
                state, active_drawer,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        print_btn.click(
            fn=commit_print,
            inputs=[paper, print_exposure, print_grade, print_contrast, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn,
                download_trigger, download_modes, dl_pkg_print, dl_pkg_both, state,
            ],
        )

        unlock_print_btn.click(
            fn=unlock_print,
            inputs=[state],
            outputs=[
                live_out, status, history,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, print_acc,
                download_trigger, download_modes, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                sample, file_in, ingest_btn,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn,
                ingest_acc, develop_acc, print_acc,
                rotate_ccw_btn, rotate_180_btn, rotate_cw_btn,
                apply_framing_btn, reset_framing_btn, auto_crop_btn, auto_straighten_btn,
                straighten_deg, crop_rect, crop_ratio,
                download_trigger, download_modes,
                state, active_drawer,
                roll_meta, roll_gallery, remove_roll_btn,
            ],
        )
        preview_tool.change(
            fn=on_preview_tool_change,
            inputs=[preview_tool],
            outputs=[live_out],
        )

        # Rail drawers are driven by #active_drawer (JS) + workflow auto-switch.
        rotate_outputs = [
            live_out, original_out, latent_out, neg_out, status, history,
            develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
            develop_acc, print_acc, state,
            straighten_deg, crop_rect, crop_ratio,
        ]
        rotate_ccw_btn.click(fn=rotate_ccw, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        rotate_cw_btn.click(fn=rotate_cw, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        rotate_180_btn.click(fn=rotate_180, inputs=[state], outputs=rotate_outputs).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )

        frame_outputs = [
            live_out, original_out, latent_out, neg_out, status, history,
            develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
            develop_acc, print_acc, state,
            straighten_deg, crop_rect, crop_ratio,
        ]
        apply_framing_btn.click(
            fn=apply_crop_straighten,
            inputs=[straighten_deg, crop_rect, crop_ratio, state],
            outputs=frame_outputs,
        ).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        reset_framing_btn.click(
            fn=reset_crop_straighten,
            inputs=[state],
            outputs=frame_outputs,
        ).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )
        auto_straighten_btn.click(
            fn=suggest_auto_straighten,
            inputs=[state],
            outputs=[straighten_deg, crop_hint],
        )
        auto_crop_btn.click(
            fn=suggest_auto_crop,
            inputs=[auto_crop_rule, crop_ratio, straighten_deg, state],
            outputs=[crop_rect, crop_hint],
        )

        curve_inputs = [
            film, developer, development_minutes, contrast,
            paper, print_grade, print_exposure, state,
        ]
        curve_outputs = [curve_plot, curve_summary]
        curve_refresh_btn.click(
            fn=refresh_curves, inputs=curve_inputs, outputs=curve_outputs
        )
        # Opening the module should show the current state, not a stale plot.
        # The JS sets #curves_open when the accordion expands.
        curves_open.change(
            fn=refresh_curves, inputs=curve_inputs, outputs=curve_outputs
        )

        spot_pos.change(fn=read_spot, inputs=[spot_pos, state], outputs=[spot_readout])

        inspect_inputs = [clip_hi, clip_lo, state]
        inspect_outputs = [hist_plot, inspect_tip, live_out, state]
        for ctrl in (clip_hi, clip_lo):
            ctrl.change(fn=refresh_inspect_tools, inputs=inspect_inputs, outputs=inspect_outputs)
        inspect_open.change(
            fn=refresh_inspect_tools, inputs=inspect_inputs, outputs=inspect_outputs
        )
        fit_tones_btn.click(
            fn=auto_fit_print_tones,
            inputs=[print_exposure, print_grade, state],
            outputs=[print_exposure, print_grade, clip_hi, clip_lo, inspect_tip, state],
        ).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        ).then(
            fn=refresh_inspect_tools,
            inputs=[clip_hi, clip_lo, state],
            outputs=[hist_plot, inspect_tip, live_out, state],
        )
        pin_ab_btn.click(
            fn=pin_ab_print, inputs=[state], outputs=[state, inspect_tip, toggle_ab_btn]
        )
        toggle_ab_btn.click(
            fn=toggle_ab_print,
            inputs=[state],
            outputs=[live_out, inspect_tip, toggle_ab_btn, state],
        )

        recipe_controls = [
            chemistry_mode, film, developer, development_minutes, contrast, grain,
            exposure_index, contrast_filter, scene_exposure, halation,
            paper, print_grade, print_exposure, print_contrast, recipe_name,
        ]
        save_recipe_btn.click(
            fn=export_recipe_file,
            inputs=recipe_controls,
            outputs=[recipe_download],
        )
        load_recipe_btn.click(
            fn=apply_recipe_file,
            inputs=[recipe_file, state],
            outputs=[
                chemistry_mode, film, developer, development_minutes, contrast, grain,
                exposure_index, contrast_filter, scene_exposure, halation,
                paper, print_grade, print_exposure, print_contrast,
                recipe_name, recipe_tip, state,
            ],
        ).then(
            fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs
        )

        # Thumbnail / button → enlarge in main preview
        swap_outputs = [
            live_out, original_out, latent_out, neg_out, inspect_out, status, state
        ]
        for slot_index, thumb in enumerate((original_out, latent_out, neg_out)):
            thumb.select(
                fn=swap_strip_slot(slot_index), inputs=[state], outputs=swap_outputs
            )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=UI_CSS,
        js=UI_JS,
    )
