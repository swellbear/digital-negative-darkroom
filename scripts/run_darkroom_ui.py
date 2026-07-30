#!/usr/bin/env python3
"""Darkroom UI: large commit-accurate live preview beside always-visible controls."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from gradio import SelectData
import numpy as np

from digital_negative.chemistry import (
    chemistry_choices,
    default_chemistry_id,
    get_chemistry,
    time_slider_bounds,
)
from digital_negative.curves import load_film_profile
from digital_negative.development import develop
from digital_negative.digital_negative import DigitalNegative
from digital_negative.display import (
    linear_to_srgb,
    negative_lightbox_preview,
    original_photo_preview,
    rotate_image,
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
from digital_negative.print_engine import print_negative

# Match commit look as closely as practical while staying interactive.
LIVE_MAX_SIDE = 2000
DRAG_MAX_SIDE = 1280  # high enough for critical judgment while dragging
INSPECT_MAX_SIDE = 3600  # high-res for zoom / inspect panel
REF_MAX_SIDE = 420

FILM_CHOICES = []
for path in list_film_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    FILM_CHOICES.append((f"{data['name']} (ISO {data['iso']})", data["id"]))

PAPER_CHOICES = []
for path in list_paper_profiles():
    data = json.loads(path.read_text(encoding="utf-8"))
    PAPER_CHOICES.append((data["name"], data["id"]))

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
.gradio-container {
  max-width: 100% !important;
  padding: 8px 12px !important;
}
/* Force controls | preview side-by-side; do not stack on typical laptop widths */
#main_workspace {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: flex-start !important;
  gap: 12px !important;
}
#main_workspace > div {
  min-width: 0 !important;
}
#controls_col {
  flex: 0 0 340px !important;
  width: 340px !important;
  max-width: 340px !important;
  position: sticky !important;
  top: 6px !important;
  /* No max-height / internal scroll — that was clipping accordion bodies
     and developer dropdown lists so options/controls never fully appeared. */
  max-height: none !important;
  overflow: visible !important;
  padding-right: 8px !important;
  align-self: flex-start !important;
  z-index: 30 !important;
}
#preview_col {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  z-index: 1 !important;
}
/* Compact control density */
#controls_col .block {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
  padding: 0 !important;
  overflow: visible !important;
}
#controls_col .label-wrap,
#controls_col label,
#controls_col .svelte-1b6s6s {
  margin-bottom: 0 !important;
  font-size: 0.82rem !important;
}
#controls_col .form {
  gap: 4px !important;
  overflow: visible !important;
}
#controls_col button {
  min-height: 34px !important;
  font-size: 0.9rem !important;
}
/* Accordion: never clip body — show full Develop/Print/Inspect controls */
#controls_col .accordion,
#controls_col .accordion > .wrap,
#controls_col .accordion > div,
#preview_col .accordion,
#preview_col .accordion > .wrap,
#preview_col .accordion > div {
  overflow: visible !important;
  max-height: none !important;
}
#controls_col .accordion .wrap,
#preview_col .accordion .wrap {
  height: auto !important;
}
#controls_col .accordion {
  margin-bottom: 6px !important;
}
#controls_col .accordion > .label-wrap {
  padding: 6px 8px !important;
  font-size: 0.9rem !important;
}
/* Open dropdown lists (film / developer / paper) must escape parents */
#controls_col [role="listbox"],
#controls_col ul.options,
#controls_col .options,
#controls_col .secondary-wrap {
  z-index: 9999 !important;
}
#controls_col [role="listbox"],
#controls_col ul.options {
  max-height: min(55vh, 420px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}
/* Avoid clipping long chemistry names in the closed dropdown */
#controls_col .wrap.fullscreen-allowed,
#controls_col .container {
  overflow: visible !important;
}
#app_header h1 {
  font-size: 1.35rem !important;
  margin: 0 0 2px 0 !important;
}
#app_header p, #app_header {
  margin: 0 0 6px 0 !important;
  font-size: 0.88rem !important;
  line-height: 1.35 !important;
}
#ritual_status {
  padding: 6px 8px !important;
  margin-bottom: 6px !important;
  border-left: 3px solid #c45c26;
  background: linear-gradient(90deg, rgba(196,92,38,0.12), transparent);
  font-size: 0.82rem !important;
  line-height: 1.3 !important;
  max-height: 7.5rem;
  overflow-y: auto;
}
#history_box {
  margin-top: 4px !important;
  padding: 6px 8px !important;
  border: 1px solid rgba(128,128,128,0.25);
  background: rgba(0,0,0,0.08);
  font-size: 0.8rem !important;
  max-height: 140px;
  overflow-y: auto;
}
#live_preview {
  min-height: 0 !important;
}
#live_preview .image-frame,
#live_preview .image-container,
#inspect_preview .image-frame,
#inspect_preview .image-container {
  overflow: auto !important;
  max-height: calc(100vh - 160px) !important;
  background: #0c0c0c !important;
}
#live_preview img,
#live_preview .image-container img,
#live_preview .image-frame img {
  max-height: calc(100vh - 160px) !important;
  width: auto !important;
  max-width: 100% !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
  transform-origin: center center;
  cursor: zoom-in;
}
#inspect_preview {
  min-height: 60vh !important;
}
#inspect_preview img,
#inspect_preview .image-container img,
#inspect_preview .image-frame img {
  max-height: none !important;
  max-width: none !important;
  width: auto !important;
  height: auto !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
  transform-origin: center center;
  cursor: grab;
}
#inspect_hint {
  font-size: 0.8rem !important;
  opacity: 0.85;
  margin: 2px 0 6px 0 !important;
}
#ref_row {
  gap: 6px !important;
  margin-top: 6px !important;
}
#ref_row img {
  max-height: 88px !important;
  object-fit: contain !important;
  background: #0c0c0c !important;
}
#db_hint {
  font-size: 0.8rem !important;
  opacity: 0.9;
  margin: 2px 0 6px 0 !important;
}
#db_size_readout {
  margin: 0 0 8px 0 !important;
  font-size: 0.85rem !important;
  color: #d8c49a;
  letter-spacing: 0.02em;
}
#db_size_readout .db-size-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  background: rgba(255, 184, 77, 0.12);
  border: 1px solid rgba(255, 184, 77, 0.35);
}
#first_print_guide button {
  min-height: 40px !important;
  font-weight: 650 !important;
}
/* Hide Gradio Timer chrome — it shows a running stopwatch and looks like
   the dodge/burn countdown never stops. We only use it as a 1s tick source. */
.db_clock_hidden {
  position: absolute !important;
  left: -9999px !important;
  top: 0 !important;
  width: 1px !important;
  height: 1px !important;
  max-height: 1px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  clip: rect(0, 0, 0, 0) !important;
  border: 0 !important;
}
.db_clock_hidden,
.db_clock_hidden * {
  visibility: hidden !important;
  font-size: 0 !important;
  line-height: 0 !important;
}
/* Gradio sometimes paints last-request duration ("9.5s") beside buttons */
#db_actions .meta-text,
#db_actions .eta-bar,
#db_actions [class*="progress"],
#db_actions [class*="duration"],
#db_actions span:has(+ button),
#controls_col .db_clock_hidden ~ .meta-text {
  display: none !important;
}
#db_flag { display: none !important; }
#db_pos { display: none !important; }
#db_wave_banner:empty,
#db_wave_banner .db-wave-idle {
  display: none !important;
}
#db_wave_banner .db-wave-active {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 0 0 8px 0;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: #1a1208;
  background: linear-gradient(90deg, #ffb84d, #ffcc66 40%, #ffe0a0 70%, #ffcc66);
  border: 1px solid #e09a30;
  box-shadow: 0 0 0 1px rgba(255, 180, 60, 0.35), 0 6px 18px rgba(0, 0, 0, 0.25);
  animation: db-wave-pulse 1.2s ease-in-out infinite;
}
#db_wave_banner .db-wave-active .db-wave-arrow {
  font-size: 1.15rem;
  opacity: 0.9;
}
body.db-exposing #controls_col {
  opacity: 0.55;
  pointer-events: none;
}
body.db-exposing #controls_col #db_actions,
body.db-exposing #controls_col .db_clock_hidden {
  pointer-events: auto;
  opacity: 1;
}
body.db-exposing #preview_col {
  position: relative;
  z-index: 2;
}
#live_preview.db-waving {
  outline: 3px solid #ffb84d !important;
  outline-offset: 4px;
  border-radius: 4px;
  box-shadow: 0 0 0 8px rgba(255, 184, 77, 0.18), 0 0 28px rgba(255, 170, 40, 0.35) !important;
  animation: db-wave-ring 1.2s ease-in-out infinite;
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
#live_preview.db-waving .label-wrap span,
#live_preview.db-waving label span {
  color: #ffb84d !important;
  font-weight: 700 !important;
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
#db_tool_cursor.db-tool-preview .db-tool-fill {
  opacity: 0.18;
}
#db_tool_cursor.db-tool-preview .db-tool-svg {
  opacity: 0.95;
}
#db_tool_cursor .db-tool-fill {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  opacity: 0.35;
  filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));
}
#db_tool_cursor .db-tool-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
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
  0%, 100% { box-shadow: 0 0 0 6px rgba(255, 184, 77, 0.16), 0 0 22px rgba(255, 170, 40, 0.28); }
  50% { box-shadow: 0 0 0 10px rgba(255, 184, 77, 0.28), 0 0 34px rgba(255, 170, 40, 0.45); }
}
@keyframes db-card-breathe {
  0%, 100% { opacity: 0.34; transform: translate(-50%, -50%) scale(0.96); }
  50% { opacity: 0.55; transform: translate(-50%, -50%) scale(1.02); }
}
@media (max-width: 900px) {
  #main_workspace { flex-wrap: wrap !important; }
  #controls_col {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    position: relative !important;
    max-height: none !important;
  }
}
"""

# Wheel / trackpad zoom + drag pan for main and inspect viewers;
# tool silhouette follows the pointer on the live print (preview + expose).
# Gradio injects launch(js=...) as a <script> text node — it must be an IIFE
# (or bare statements), not a bare () => {} which never runs.
UI_JS = """
(() => {
  window.__dbPos = '';
  window.__dbGetPos = () => window.__dbPos || '';
  window.__dbToolArmed = true;
  window.__dbToolScale = 1.0;

  const clampToolScale = (s) => Math.min(2.75, Math.max(0.35, s));

  const updateSizeReadout = () => {
    const el = document.querySelector('#db_size_readout .db-size-value');
    if (!el) return;
    const pct = Math.round(clampToolScale(window.__dbToolScale || 1) * 100);
    el.textContent = pct + '%';
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
      img.style.cursor = scale > 1.02 ? (dragging ? 'grabbing' : 'grab') : 'zoom-in';
    };

    root.addEventListener('wheel', (e) => {
      const img = findImg();
      if (!img) return;
      // Live print: scroll = resize dodge/burn tool. Ctrl/Meta+scroll = zoom.
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

    root.addEventListener('pointerdown', (e) => {
      if (root.classList.contains('db-waving')) return;
      const img = findImg();
      if (!img || scale <= 1.02) return;
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      root.setPointerCapture?.(e.pointerId);
      apply(img);
    });
    root.addEventListener('pointermove', (e) => {
      if (!dragging || root.classList.contains('db-waving')) return;
      const img = findImg();
      if (!img) return;
      panX += e.clientX - lastX;
      panY += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
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

  const forceNoCursor = (live) => {
    if (!live) return;
    live.style.cursor = 'none';
    live.querySelectorAll('*').forEach((el) => { el.style.cursor = 'none'; });
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
    if (live) live.classList.remove('db-tool-hover');
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
    const stroke = (flag.mode || '').toLowerCase().startsWith('dodge') ? '#66ccff' : '#ffcc66';
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
    const flag = readFlag();
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
    showToolAt(e.clientX, e.clientY, flag, false);
  };

  let bootScheduled = false;
  const boot = () => {
    // Coalesce observer storms — never re-enter synchronously.
    if (bootScheduled) return;
    bootScheduled = true;
    requestAnimationFrame(() => {
      bootScheduled = false;
      if (window.__dbBootLock) return;
      window.__dbBootLock = true;
      try {
        enhance('#live_preview');
        enhance('#inspect_preview');
        hideClockChrome();
        syncWave();
      } finally {
        window.__dbBootLock = false;
      }
    });
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
    const specs = [
      ['#db_flag', { childList: true, subtree: true, attributes: true, attributeFilter: ['data-exposing', 'data-shape', 'data-mode', 'data-stamp-fw', 'src'] }],
      ['#live_preview', { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'class'] }],
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


def _chem_time_update(film_id: str, developer_id: str, *, reset_to_normal: bool = True):
    """Gradio updates for developer dropdown / minutes slider."""
    profile = _film_profile(film_id)
    chem = get_chemistry(profile, developer_id)
    if chem is None:
        label = "Dev time (rel. ×8 min stand-in)"
        return gr.update(minimum=4.0, maximum=16.0, value=8.0, label=label, step=0.25)
    tmin, tmax, normal = time_slider_bounds(chem)
    family = chem.get("curve_family") or []
    if isinstance(family, list) and len(family) >= 2:
        times = ", ".join(f"{float(m['minutes']):g}" for m in sorted(family, key=lambda x: x["minutes"]))
        label = f"Dev time (min) · N={normal:g} · family [{times}]"
    else:
        label = f"Dev time (min) · N={normal:g} @ 20°C (morph)"
    return gr.update(minimum=tmin, maximum=tmax, value=normal, label=label, step=0.25)


def on_film_change(film_id: str):
    profile = _film_profile(film_id)
    chem_id = default_chemistry_id(profile)
    choices = chemistry_choices(profile)
    return (
        gr.update(choices=choices, value=chem_id),
        _chem_time_update(film_id, chem_id, reset_to_normal=True),
    )


def on_developer_change(film_id: str, developer_id: str):
    return _chem_time_update(film_id, developer_id, reset_to_normal=True)


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
    upload_path = None
    if file_obj is not None:
        if isinstance(file_obj, str):
            upload_path = file_obj
        else:
            upload_path = getattr(file_obj, "name", None) or str(file_obj)
        if upload_path:
            return upload_path
    if sample_path:
        return sample_path
    return None


def _to_rgb_u8(gray_float: np.ndarray, *, assume_linear: bool = False) -> np.ndarray:
    view = linear_to_srgb(gray_float) if assume_linear else gray_float
    g = to_u8_gray(view)
    return np.stack([g, g, g], axis=-1)


def _downscale_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return np.ascontiguousarray(rgb[::step, ::step])


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


def _history_md(dn) -> str:
    hist = dn.metadata.get("history", [])
    lines = ["### Decision log", "_Locked decisions only — exploring does not write here._", ""]
    if not hist:
        lines.append("_No locked decisions yet. Commit a stage to record it._")
    for i, h in enumerate(hist, 1):
        op = h.get("op", "?")
        if op == "ingest":
            lines.append(f"{i}. **Ingest** — `{h.get('source')}`")
        elif op == "develop":
            chem = h.get("developer_name") or h.get("developer_id")
            if h.get("development_minutes") is not None:
                time_bit = f"{float(h['development_minutes']):g} min"
            else:
                time_bit = f"rel={h.get('relative_time')}"
            lines.append(
                f"{i}. **Develop** — `{h.get('film_profile_id')}` · {chem} · "
                f"{time_bit} · N±={h.get('contrast_modifier')} · "
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
            lines.append(
                f"{i}. **Print** — `{h.get('paper_id')}` · grade {h.get('grade')} · "
                f"exp {exp_bit}"
                + (f" · nudge={h.get('contrast')}" if h.get("contrast") not in (None, 0, 0.0) else "")
                + db_bit
            )
        elif op == "unlock":
            stage = h.get("stage", "?")
            label = {"development": "Develop", "print": "Print", "ingest": "Ingest"}.get(stage, stage)
            lines.append(f"{i}. **← Unlocked {label}** — previous lock opened for revision")
        elif op == "rotate":
            lines.append(
                f"{i}. **Rotate** — {h.get('degrees_cw'):+g}° "
                f"(total {h.get('total_degrees')}° CW)"
            )
        else:
            lines.append(f"{i}. **{op}**")
    locks = dn.metadata.get("ui_state", {}).get("locked_stages", [])
    lock_labels = []
    for s in ("ingest", "development", "print"):
        if s in locks:
            lock_labels.append({"ingest": "Ingest", "development": "Develop", "print": "Print"}[s])
    lines.append("")
    lines.append(
        f"**Currently locked:** {', '.join(lock_labels) or '—'}  \n"
        f"**Process seed:** `{dn.metadata.get('process_seed')}` "
        f"_(mild tank variation; same seed = repeatable)_"
    )
    return "\n".join(lines)


def _stage_banner(stage: str, locked: list | None = None) -> str:
    """Ritual progress: which stage you're working, which are locked."""
    steps = [("ingest", "Ingest"), ("development", "Develop"), ("print", "Print")]
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
    if "### Decision log" in full:
        status, hist = full.split("### Decision log", 1)
        return status.strip(), "### Decision log" + hist
    if "### Decision history" in full:
        status, hist = full.split("### Decision history", 1)
        return status.strip(), "### Decision log" + hist
    return full, ""


_VIEWER_LABELS = {
    "live": "Commit preview (live) — theoretical print",
    "original": "Original photo (enlarged) — click Live print to return",
    "latent": "Latent DN (enlarged) — click Live print to return",
    "negative": "Developed negative (enlarged) — click Live print to return",
}


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
        img = live if live is not None else (state or {}).get("live_rgb")
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


def _pack_preview(live, original, latent, neg, summary, state):
    status, hist = _split_summary(summary or "")
    if state is not None:
        if live is not None:
            state = {**state, "live_rgb": live, "live_inspect": live}
        if neg is not None:
            state = {
                **state,
                "neg_ref": neg,
                "neg_view": neg,
                "neg_inspect": state.get("neg_inspect")
                if state.get("neg_inspect") is not None
                else neg,
            }
    shown = _viewer_frame(state, live=live, original=original, latent=latent, neg=neg)
    return shown, original, latent, neg, status, hist, _inspect_frame(state, live=live), state


def focus_viewer(mode: str):
    """Return a handler that puts a reference (or live print) in the large preview."""

    def _fn(state, evt: SelectData | None = None):
        if not state or state.get("dn") is None:
            empty = gr.update()
            return empty, empty, "*Commit Ingest first.*", gr.update(open=False), state
        if evt is not None and getattr(evt, "selected", True) is False:
            mode_use = "live"
        else:
            mode_use = mode
        state = {**state, "viewer_mode": mode_use}
        tip = {
            "live": "_Large + Inspect: **live theoretical print**. Scroll-wheel zooms; drag pans._",
            "original": "_Large + Inspect: **Original**. Scroll-wheel zooms; drag pans._",
            "latent": "_Large + Inspect: **Latent DN**. Scroll-wheel zooms; drag pans._",
            "negative": "_Large + Inspect: **Developed negative**. Scroll-wheel zooms; drag pans._",
        }.get(mode_use, "")
        banner = _stage_banner(state.get("stage", "development"), _locks(state))
        status = f"{banner}\n\n{tip}"
        return (
            _viewer_frame(state),
            _inspect_frame(state),
            status,
            gr.update(open=True),
            state,
        )

    return _fn


def focus_viewer_button(mode: str):
    def _fn(state):
        return focus_viewer(mode)(state, None)

    return _fn


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


def rotate_working(turns_cw: int, state):
    """Rotate the Digital Negative and reference previews; clears Develop/Print locks."""
    if not state or state.get("dn") is None:
        raise gr.Error("Commit Ingest first.")

    dn = state["dn"]
    dn.image = rotate_image(dn.image, turns_cw)
    ingest = dn.metadata.setdefault("ingest", {})
    degrees = int(ingest.get("rotation_degrees", 0)) + (90 * int(turns_cw))
    ingest["rotation_degrees"] = degrees % 360

    for key in (
        "original_view",
        "original_ref",
        "original_inspect",
        "latent_view",
        "latent_ref",
        "latent_inspect",
        "live_rgb",
        "live_inspect",
    ):
        if state.get(key) is not None:
            state[key] = rotate_image(state[key], turns_cw)

    ui = dn.metadata.setdefault("ui_state", {})
    locks = ui.setdefault("locked_stages", [])
    committed = ui.setdefault("committed_stages", [])
    for stage in ("print", "development"):
        if stage in locks:
            locks.remove(stage)
        if stage in committed:
            committed.remove(stage)
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
            "stage": "development",
            "original_view": state.get("original_view"),
            "original_ref": state.get("original_ref"),
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
    )


def rotate_cw(state):
    return rotate_working(1, state)


def rotate_ccw(state):
    return rotate_working(-1, state)


def rotate_180(state):
    return rotate_working(2, state)


def commit_ingest(sample_path, file_obj, state):
    path = _resolve_input(file_obj, sample_path)
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
        f"**Ingest locked** — `{dn.metadata['source']['original_filename']}`  \n"
        f"_Open a sequence below, then use **Inspect & zoom** (scroll-wheel zoom, drag pan)._\n\n"
        f"{_history_md(dn)}"
    )
    state = {
        "dn": dn,
        "proxy": _proxy_dn(dn, LIVE_MAX_SIDE),
        "proxy_drag": _proxy_dn(dn, DRAG_MAX_SIDE),
        "original_ref": original_ref,
        "original_view": original_view,
        "original_inspect": original_inspect,
        "latent_ref": latent_ref,
        "latent_view": latent_view,
        "latent_inspect": latent_inspect,
        "neg_ref": None,
        "neg_view": None,
        "neg_inspect": None,
        "live_rgb": latent_view,
        "live_inspect": latent_inspect,
        "viewer_mode": "live",
        "development": None,
        "development_full": None,
        "stage": "development",
        "summary_cache": summary,
        "source_path": path,
    }
    return (
        gr.update(value=latent_view, label=_VIEWER_LABELS["live"]),
        original_ref,
        latent_ref,
        None,
        *_split_summary(summary),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(open=False),  # collapse Ingest
        gr.update(open=True),   # show Develop
        gr.update(open=True),   # keep Print reachable
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        _inspect_frame(state, live=latent_inspect),
        gr.update(open=True),
        state,
    )


def _run_live_develop_then_print(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
    *,
    max_side: int = LIVE_MAX_SIDE,
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
        commit=False,
    )
    paper = load_paper_profile(_profile_path(list_paper_profiles(), paper_id))
    printed = print_negative(
        development.transmittance,
        state["dn"],
        paper,
        base_exposure_seconds=float(print_exposure),
        grade=float(print_grade),
        contrast=float(print_contrast),
        local_stops=local_stops_from_state(state),
        commit=False,
    )
    live_rgb = _to_rgb_u8(printed.preview)
    neg_full = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    speed = state["dn"].metadata.get("print", {}).get("filtration", {}).get("values", {}).get(
        "filter_speed", 1.0
    )
    quality_note = "drag" if max_side <= DRAG_MAX_SIDE else "hq"
    curve_src = proxy.metadata.get("development", {}).get("curve_source", "?")
    summary = (
        f"{_stage_banner('development', _locks(state))}\n\n"
        f"**Live print** {live_rgb.shape[1]}×{live_rgb.shape[0]} ({quality_note})  \n"
        f"{profile.name} · {developer_id} · {float(development_minutes):g} min · "
        f"curve={curve_src} · N±={float(contrast):+.2f} · grain={float(grain):.2f}  \n"
        f"{paper.name} · g{float(print_grade):.1f} · {_print_timer_label(print_exposure)} "
        f"· ×{float(speed):.2f}\n\n"
        f"{_history_md(state['dn'])}"
    )
    state = {
        **state,
        "proxy": state.get("proxy") or _proxy_dn(state["dn"], LIVE_MAX_SIDE),
        "proxy_drag": state.get("proxy_drag") or _proxy_dn(state["dn"], DRAG_MAX_SIDE),
        "development": development,
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
        },
    }
    state = _remember_print_seconds(state, print_exposure)
    return live_rgb, neg_ref, summary, state


def live_preview(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
    quality: str = "high",
):
    """Unified live viewer: develop+print while developing; print-only after Develop lock.

    quality='drag' uses a faster lower-res proxy while sliders move;
    quality='high' (release / change) uses commit-accurate resolution.
    """
    max_side = DRAG_MAX_SIDE if quality == "drag" else LIVE_MAX_SIDE

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
                )
            if max(t.shape) > max_side:
                step = max(1, int(np.ceil(max(t.shape) / max_side)))
                t = np.ascontiguousarray(t[::step, ::step])

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
        state = {**state, "print_draft": result, "live_rgb": live_rgb, "summary_cache": summary}
        state = _remember_print_seconds(state, print_exposure)
        return _pack_preview(
            live_rgb,
            state.get("original_ref"),
            state.get("latent_ref"),
            state.get("neg_ref"),
            summary,
            state,
        )

    # Develop unlocked: show print through the working negative
    live_rgb, neg_ref, summary, state = _run_live_develop_then_print(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        max_side=max_side,
    )
    return _pack_preview(
        live_rgb, state.get("original_ref"), state.get("latent_ref"), neg_ref, summary, state
    )


def live_preview_drag(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
):
    return live_preview(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        quality="drag",
    )


def live_preview_high(
    film_id,
    developer_id,
    development_minutes,
    contrast,
    grain,
    paper_id,
    print_exposure,
    print_grade,
    print_contrast,
    state,
):
    return live_preview(
        film_id,
        developer_id,
        development_minutes,
        contrast,
        grain,
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
        state,
        quality="high",
    )


def commit_develop(film_id, developer_id, development_minutes, contrast, grain, state):
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
        commit=True,
    )
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "development" not in locks:
        locks.append("development")

    step = max(1, int(np.ceil(max(development.transmittance.shape) / LIVE_MAX_SIDE)))
    t_proxy = np.ascontiguousarray(development.transmittance[::step, ::step])

    # Keep last theoretical print on screen until .then refreshes with Print controls;
    # fall back to positive if no live print was generated yet.
    # Do not use `or` — live_rgb is a numpy array (ambiguous truth value).
    live_view = state.get("live_rgb")
    if live_view is None:
        live_view = _downscale_rgb(
            _to_rgb_u8(development.positive_preview), LIVE_MAX_SIDE
        )
    neg_full = _to_rgb_u8(negative_lightbox_preview(development.transmittance))
    neg_inspect = _downscale_rgb(neg_full, INSPECT_MAX_SIDE)
    neg_view = _downscale_rgb(neg_full, LIVE_MAX_SIDE)
    neg_ref = _downscale_rgb(neg_full, REF_MAX_SIDE)
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Develop locked** — refine Print below, then Commit Print.\n\n{_history_md(dn)}"
    )
    state = {
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
        "stage": "print",
        "summary_cache": summary,
        "source_path": state.get("source_path"),
        "db_accum": None,
        "db_exposing": False,
        "db_seconds_left": 0,
        "db_strokes": [],
    }
    return (
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
    )


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
    )
    stages = dn.metadata["ui_state"].setdefault("committed_stages", [])
    locks = dn.metadata["ui_state"].setdefault("locked_stages", [])
    if "print" not in stages:
        stages.append("print")
    if "print" not in locks:
        locks.append("print")

    live_rgb = _downscale_rgb(_to_rgb_u8(result.preview), LIVE_MAX_SIDE)
    speed = dn.metadata["print"]["filtration"]["values"].get("filter_speed", 1.0)
    db_note = f" · {len(strokes)} dodge/burn pass(es)" if strokes else ""
    summary = (
        f"{_stage_banner('print', locks)}\n\n"
        f"**Print locked** — {paper.name} · g{float(print_grade):.1f} · "
        f"{_print_timer_label(print_exposure)}{db_note}\n\n{_history_md(dn)}"
    )
    state = {**state, "print": result, "live_rgb": live_rgb, "summary_cache": summary}
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
        state,
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
            inspect_out,
            inspect_acc_u,
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
        inspect_out = state.get("live_inspect") or live_rgb
        inspect_acc_u = gr.update(open=True)

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
        paper_id,
        print_exposure,
        print_grade,
        print_contrast,
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
        inspect_out,
        inspect_acc_u,
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
        state,
    )


def reset_session():
    summary = (
        "**1. Ingest — working** → 2. Develop → 3. Print\n\n"
        "*Commit Ingest to begin.*"
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
        None,
    )


def build_ui() -> gr.Blocks:
    default_sample = SAMPLE_CHOICES[1][1] if len(SAMPLE_CHOICES) > 1 else ""
    with gr.Blocks(title="Digital Negative Darkroom") as demo:
        state = gr.State(None)
        gr.Markdown(
            """
            # Digital Negative Darkroom
            **Ingest → Develop → Print** · Commit locks · Unlock revises · Right image = enlarger easel / theoretical print
            """,
            elem_id="app_header",
        )
        with gr.Accordion("First print (≈2 min)", open=True, elem_id="first_print_guide"):
            gr.Markdown(
                "New here? **Run first-print guide** locks Ingest + Develop with the current "
                "sample, opens Print with a soft-oval card, and leaves you on the easel. "
                "Then scroll to size the card and **Start — wave over print →**."
            )
            guide_btn = gr.Button(
                "▶ Run first-print guide",
                variant="primary",
                size="sm",
            )

        with gr.Row(elem_id="main_workspace", equal_height=False):
            with gr.Column(scale=0, elem_id="controls_col", min_width=300):
                status = gr.Markdown(
                    "**1. Ingest — working** → 2. Develop → 3. Print  \n"
                    "_Commit Ingest to begin._",
                    elem_id="ritual_status",
                )
                with gr.Accordion("1 · Ingest", open=True) as ingest_acc:
                    sample = gr.Dropdown(
                        choices=SAMPLE_CHOICES,
                        value=default_sample,
                        label="Sample (if no upload)",
                    )
                    file_in = gr.File(
                        label="Upload (overrides sample)",
                        file_types=[
                            ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2",
                            ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp",
                            ".heic", ".heif", ".avif",
                        ],
                        # Gradio 6 needs a real drop-zone height — 56px made "Click to upload" unreliable.
                        height=120,
                        elem_id="ingest_upload",
                    )
                    ingest_btn = gr.Button("Commit Ingest", variant="primary", size="sm")

                # Keep later stages collapsed until Commit Ingest / Commit Develop open them.
                with gr.Accordion("2 · Develop", open=False) as develop_acc:
                    film = gr.Dropdown(
                        choices=FILM_CHOICES,
                        value=FILM_CHOICES[0][1] if FILM_CHOICES else None,
                        label="Film",
                    )
                    developer = gr.Dropdown(
                        choices=_INIT_DEV_CHOICES,
                        value=_INIT_DEV_ID,
                        label="Developer",
                    )
                    development_minutes = gr.Slider(
                        _INIT_TMIN,
                        _INIT_TMAX,
                        value=_INIT_TNORM,
                        step=0.25,
                        label=f"Dev time (min) · N={_INIT_TNORM:g} @ 20°C",
                    )
                    contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Contrast N− / N+")
                    grain = gr.Slider(0.0, 2.5, value=1.0, step=0.05, label="Grain")
                    with gr.Row():
                        develop_btn = gr.Button(
                            "Commit Develop", interactive=False, variant="primary", size="sm"
                        )
                        unlock_develop_btn = gr.Button("Unlock", interactive=False, size="sm")

                with gr.Accordion("3 · Print", open=False) as print_acc:
                    paper = gr.Dropdown(
                        choices=PAPER_CHOICES,
                        value=PAPER_CHOICES[0][1] if PAPER_CHOICES else None,
                        label="Paper",
                    )
                    print_exposure = gr.Slider(
                        2.0,
                        64.0,
                        value=8.0,
                        step=0.5,
                        label="Base exposure (seconds)",
                        info="Enlarger timer — dodge/burn is relative to this base",
                    )
                    base_math_md = gr.Markdown(_base_math_md(8.0), elem_id="base_math")
                    print_grade = gr.Slider(0.0, 5.0, value=2.5, step=0.5, label="MG filtration")
                    print_contrast = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Filter nudge")
                    with gr.Accordion("Dodge & burn · enlarger easel", open=True):
                        gr.Markdown(
                            "The **print on the right** is the easel. Pick a card shape — outline "
                            "follows your pointer. **Scroll** over the print to resize the tool. "
                            "Burn adds enlarger light (**darker**); dodge holds light back (**lighter**). "
                            "Start, then wave. Reset clears local work only.",
                            elem_id="db_hint",
                        )
                        db_shape = gr.Radio(
                            choices=[(label, key) for key, label in CARD_PRESETS],
                            value="soft_oval",
                            label="Card / wand shape",
                        )
                        db_editor = gr.ImageEditor(
                            label="Custom card (paint only if shape = Custom)",
                            type="numpy",
                            image_mode="RGBA",
                            height=220,
                            value=tool_workshop_canvas(),
                            brush=gr.Brush(
                                default_size=48,
                                colors=["#ffcc66", "#ffffff", "#66ccff"],
                                default_color="#ffcc66",
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
                                ("Dodge — hold back light (lighter print)", "dodge"),
                                ("Burn — add enlarger light (darker print)", "burn"),
                            ],
                            value="burn",
                            label="Mode",
                        )
                        db_seconds = gr.Slider(
                            1,
                            32,
                            value=4,
                            step=1,
                            label="Dodge/burn pass (seconds)",
                            info="Relative to the base exposure timer above",
                        )
                        pass_math_md = gr.Markdown(_pass_math_md(8.0, 4.0, "burn"), elem_id="pass_math")
                        db_timer_md = gr.Markdown(
                            "**Ready** — pick a shape, then **Start — wave over print →**"
                        )
                        with gr.Row(elem_id="db_actions"):
                            db_start_btn = gr.Button(
                                "Start — wave over print →",
                                variant="primary",
                                size="sm",
                            )
                            db_reset_btn = gr.Button("Reset local work", size="sm")
                        db_flag = gr.HTML(_db_flag_html(None), elem_id="db_flag")
                        # Keep in the DOM (CSS-hidden) so Timer.tick + pointer JS can sync it.
                        db_pos = gr.Textbox(value="0.5000,0.5000", elem_id="db_pos", show_label=False)
                        # Off-screen tick source — samples the waved card ~4×/sec.
                        with gr.Column(elem_classes=["db_clock_hidden"]):
                            db_clock = gr.Timer(value=TICK_SECONDS, active=False)
                    with gr.Row():
                        print_btn = gr.Button(
                            "Commit Print", interactive=False, variant="primary", size="sm"
                        )
                        unlock_print_btn = gr.Button("Unlock", interactive=False, size="sm")

                reset_btn = gr.Button("New negative", size="sm")
                with gr.Accordion("Decision log", open=False):
                    history = gr.Markdown(
                        "_Locked decisions only — exploring does not write here._",
                        elem_id="history_box",
                    )

            with gr.Column(scale=1, elem_id="preview_col", min_width=480):
                db_wave_banner = gr.HTML(_wave_banner_html(None), elem_id="db_wave_banner")
                db_size_readout = gr.HTML(
                    '<div class="db-size-pill">Card size <strong class="db-size-value">100%</strong> · scroll over print to resize · Ctrl/⌘+scroll zooms</div>',
                    elem_id="db_size_readout",
                )
                live_out = gr.Image(
                    label=LIVE_PRINT_LABEL,
                    type="numpy",
                    elem_id="live_preview",
                    height=620,
                    buttons=["fullscreen", "download"],
                )
                with gr.Row():
                    rotate_ccw_btn = gr.Button("Rotate ⟲ 90°", size="sm", interactive=False)
                    rotate_180_btn = gr.Button("Rotate 180°", size="sm", interactive=False)
                    rotate_cw_btn = gr.Button("Rotate 90° ⟳", size="sm", interactive=False)
                with gr.Row(elem_id="ref_row"):
                    original_out = gr.Image(
                        label="Original (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                    latent_out = gr.Image(
                        label="Latent DN (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                    neg_out = gr.Image(
                        label="Negative (click to enlarge)",
                        type="numpy",
                        height=96,
                        buttons=["fullscreen"],
                    )
                with gr.Row():
                    view_orig_btn = gr.Button("Original", size="sm")
                    view_lat_btn = gr.Button("Latent DN", size="sm")
                    view_neg_btn = gr.Button("Negative", size="sm")
                    view_live_btn = gr.Button("Live print", size="sm", variant="primary")
                with gr.Accordion("Inspect & zoom", open=True) as inspect_acc:
                    gr.Markdown(
                        "Scroll-wheel zooms · drag pans when zoomed · double-click resets · "
                        "fullscreen button for an even larger view",
                        elem_id="inspect_hint",
                    )
                    inspect_out = gr.Image(
                        label="Inspect — pick a sequence above",
                        type="numpy",
                        elem_id="inspect_preview",
                        height=720,
                        buttons=["fullscreen", "download"],
                    )

        # Always pass develop + print controls so the large viewer can show a
        # theoretical print through the working negative while developing.
        preview_inputs = [
            film,
            developer,
            development_minutes,
            contrast,
            grain,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
            state,
        ]
        preview_outputs = [
            live_out, original_out, latent_out, neg_out, status, history, inspect_out, state
        ]

        ingest_btn.click(
            fn=commit_ingest,
            inputs=[sample, file_in, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                sample, file_in, ingest_btn, develop_btn, print_btn,
                ingest_acc, develop_acc, print_acc,
                rotate_ccw_btn, rotate_180_btn, rotate_cw_btn,
                inspect_out, inspect_acc, state,
            ],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        for ctrl in (
            development_minutes,
            contrast,
            grain,
            paper,
            print_exposure,
            print_grade,
            print_contrast,
        ):
            # Drag = fast lower-res; release/change = commit-quality preview
            ctrl.input(fn=live_preview_drag, inputs=preview_inputs, outputs=preview_outputs)
            ctrl.change(fn=live_preview_high, inputs=preview_inputs, outputs=preview_outputs)

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
            outputs=[developer, development_minutes],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        developer.change(
            fn=on_developer_change,
            inputs=[film, developer],
            outputs=[development_minutes],
        ).then(
            fn=live_preview_high,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )

        develop_btn.click(
            fn=commit_develop,
            inputs=[film, developer, development_minutes, contrast, grain, state],
            outputs=[
                live_out, original_out, latent_out, neg_out, status, history,
                film, developer, development_minutes, contrast, grain,
                develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
                develop_acc, print_acc, state,
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

        guide_btn.click(
            fn=guided_first_print,
            inputs=[
                sample,
                file_in,
                film,
                developer,
                development_minutes,
                contrast,
                grain,
                paper,
                print_exposure,
                print_grade,
                print_contrast,
                state,
            ],
            outputs=[
                live_out,
                original_out,
                latent_out,
                neg_out,
                status,
                history,
                sample,
                file_in,
                ingest_btn,
                film,
                developer,
                development_minutes,
                contrast,
                grain,
                develop_btn,
                unlock_develop_btn,
                print_btn,
                unlock_print_btn,
                ingest_acc,
                develop_acc,
                print_acc,
                rotate_ccw_btn,
                rotate_180_btn,
                rotate_cw_btn,
                inspect_out,
                inspect_acc,
                state,
                db_editor,
                db_timer_md,
                base_math_md,
                pass_math_md,
                db_shape,
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
                develop_acc, print_acc, state,
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
                print_btn, unlock_print_btn, state,
            ],
        )

        unlock_print_btn.click(
            fn=unlock_print,
            inputs=[state],
            outputs=[
                live_out, status, history,
                paper, print_exposure, print_grade, print_contrast,
                print_btn, unlock_print_btn, print_acc, state,
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
                rotate_ccw_btn, rotate_180_btn, rotate_cw_btn, state,
            ],
        )

        rotate_outputs = [
            live_out, original_out, latent_out, neg_out, status, history,
            develop_btn, unlock_develop_btn, print_btn, unlock_print_btn,
            develop_acc, print_acc, state,
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

        # Thumbnail / button → enlarge in main preview + open inspect/zoom
        focus_outputs = [live_out, inspect_out, status, inspect_acc, state]
        original_out.select(fn=focus_viewer("original"), inputs=[state], outputs=focus_outputs)
        latent_out.select(fn=focus_viewer("latent"), inputs=[state], outputs=focus_outputs)
        neg_out.select(fn=focus_viewer("negative"), inputs=[state], outputs=focus_outputs)
        view_orig_btn.click(fn=focus_viewer_button("original"), inputs=[state], outputs=focus_outputs)
        view_lat_btn.click(fn=focus_viewer_button("latent"), inputs=[state], outputs=focus_outputs)
        view_neg_btn.click(fn=focus_viewer_button("negative"), inputs=[state], outputs=focus_outputs)
        view_live_btn.click(fn=focus_viewer_button("live"), inputs=[state], outputs=focus_outputs)
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
