# Calibration Notes (fill after side-by-side sessions)

**Purpose:** Capture the 2–3 highest-leverage process-feel adjustments so they can be implemented as targeted code/profile changes — not a laundry list.

**How to use:** After comparing a real raw → this app vs film + darkroom print (or a close proxy), fill one section per finding. Prefer concrete, observable language (“push +1 still too flat in the sky”) over vague (“contrast feels off”).

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
| Film profile in app | e.g. Color Neg 400 (spectral) |
| Real-world reference | color-neg scan / RA-4 print / slide |
| Lighting notes | |

**What to judge first**

- Orange mask / lightbox invert of C-41 negatives (does skin/sky polarity feel right?)
- Push +1 vs Pull −1 on C-41 (contrast and grain, not just density)
- RA-4 CC filtration: does adding Magenta / Cyan / Yellow move the print the expected way?
- E-6 slide finish: positive polarity, saturation, highlight roll-off
- Confirm B&W Chemistry mode is unchanged after color work

**Where to change when calibrating**

- Film layer curves / mask / interimage → `profiles/films/*spectral*.json`
- Dye peaks / sensitivity widths → same profile `spectral.layers`
- RA-4 paper toe/shoulder / dye peaks → `profiles/papers/ra4-glossy-v1.json`
- Engine behavior → `src/digital_negative/color_development.py`, `color_print.py`, `spectral.py`
