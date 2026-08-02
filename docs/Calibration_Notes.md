# Calibration Notes (fill after side-by-side sessions)

**Purpose:** Capture the 2–3 highest-leverage process-feel adjustments so they can be implemented as targeted code/profile changes — not a laundry list.

**How to use:** After comparing a real raw → this app vs film + darkroom print (or a close proxy), fill one section per finding. Prefer concrete, observable language (“push +1 still too flat in the sky”) over vague (“contrast feels off”).

**Parked:** automated scan-corpus / ML calibration helpers — see Critique_Roadmap “Parked”; stick to hand side-by-sides for now.

---

## Session meta

| Field | Value |
|-------|-------|
| Date | |
| Scene / raw file | |
| Film stock in app | |
| Real-world reference | (film + paper / scan / memory) |
| Lighting notes | |

---

## Finding 1 (highest leverage)

- **Control / region:** (e.g. relative development push, N+, grade 4, HP5 toe, grain midtones)
- **What you see in the app:**
- **What you expect from the darkroom:**
- **Proposed change:** (direction + magnitude if known — e.g. “more highlight density on push >1.3”)
- **Where to change:** `curves.py` / film JSON / `print_engine.py` / paper JSON / grain — if known

---

## Finding 2

- **Control / region:**
- **What you see in the app:**
- **What you expect from the darkroom:**
- **Proposed change:**
- **Where to change:**

---

## Finding 3

- **Control / region:**
- **What you see in the app:**
- **What you expect from the darkroom:**
- **Proposed change:**
- **Where to change:**

---

## Explicit non-goals this cycle

_(Things that looked fine — do not touch)_

-

---

## After implementation

- [ ] Re-check the same scene in the live UI
- [ ] Confirm no regression on pull / soft grade / Standard developer
- [ ] Update Critique_Roadmap.md status if this cycle closes

---

## Color chemistry targets (spectral C-41 / E-6 / RA-4)

Use the same side-by-side discipline as B&W once Color Chemistry mode is active.

| Field | Value |
|-------|-------|
| Date | |
| Scene / raw file | |
| Color path | C-41 → RA-4 / E-6 slide |
| Film profile in app | e.g. Kodak Portra 400 / Ektachrome E100 / Provia 100F |
| Real-world reference | color-neg scan / RA-4 print / slide |
| Lighting notes | |

**What to judge first**

- Orange mask / lightbox invert of C-41 negatives (does skin/sky polarity feel right?)
- Push +1 vs Pull −1 on C-41 (contrast and grain, not just density)
- RA-4 CC filtration: does adding Magenta / Cyan / Yellow move the print the expected way?
- E-6 slide finish: positive polarity, saturation, highlight roll-off
- Confirm B&W Chemistry mode is unchanged after color work
- Named catalog differentiation: Portra 160 vs 400 vs 800; E100 vs Provia vs Velvia; T-Max 400 vs Tri-X; Delta 400 vs HP5; Acros vs Delta 100

**Where to change when calibrating**

- Film layer curves / mask / interimage → `profiles/films/*spectral*.json`
- Dye peaks / sensitivity widths → same profile `spectral.layers`
- RA-4 paper toe/shoulder / dye peaks → `profiles/papers/ra4-glossy-v1.json`
- Engine behavior → `src/digital_negative/color_development.py`, `color_print.py`, `spectral.py`

---

## Named film catalog (Kodak / Ilford / Fuji) — approximate character limits

First named batch (2026-07-31). All stocks document public datasheet sources in each profile `source{}`.

| Profile | Quality of curve data |
|---------|----------------------|
| Tri-X 400 D-76 / T-MAX | True multi-time `curve_family` (F-4017) |
| Other Tri-X chems, HP5 / FP4 / Delta 100 | Single curve + CI×time morph |
| T-Max 400, Delta 400, Acros 100 II | New single-curve approx + datasheet times (morph); no multi-time families yet |
| Portra 160 / 400 / 800, E100, Provia 100F, Velvia 50 | Authored spectral / logistic layer curves inspired by published process aims — **not** licensed Kodak/Fujifilm digitizations or LUTs |

Prefer densitometer or WebPlotDigitizer re-digitization of official plots before a second large catalog expansion.
