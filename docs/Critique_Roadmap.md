# Next Critique Steps — Toward the End Goal

**Date:** 2026-07-30  
**Repository:** https://github.com/swellbear/digital-negative-darkroom  

The project has moved past architectural and basic UX hurdles. Live preview, stage locks, unlock/revise, and a working B&W pipeline are in place. Remaining work is less about “building the machine” and more about making it *feel* like a real darkroom and turning it into something photographers will actually want to use.

---

## 1. Process Feel Calibration (Highest Priority)

This is the single biggest remaining gap versus the original vision.

**What to critique next:**
- Side-by-side tests: real raw → Digital Negative Darkroom result vs. actual film + darkroom print of the same scene (or a close proxy).
- Does relative development + contrast (N±) behave the way you expect when pushing/pulling?
- Do the developer styles (Standard / High Definition / High Energy) produce distinct, believable character differences?
- Does multigrade grade + filter speed + exposure feel natural when you change them while looking at the live print preview?
- Grain: is the strength and character appropriate across the tonal range?

**Goal of this critique cycle:**  
Identify the **2–3 highest-leverage** curve/modifier adjustments that would make the results feel more authentic to your eye. Document them in [`Calibration_Notes.md`](Calibration_Notes.md) so they can be implemented as targeted changes.

---

## 2. Ritual Experience Depth

The live + Commit/Unlock model is good. Now refine the *experience* of the ritual.

**Critique questions:**
- Is the visual hierarchy clear about which stage is active vs locked?
- Does the history panel feel useful or just decorative?
- When you unlock a stage, is the transition clean and understandable?
- Does the mild process variation add life without becoming distracting or unpredictable?
- Would a photographer who has spent hundreds of hours in a real darkroom feel the sequential decision-making, or does it still feel like a conventional editor with extra buttons?

---

## 3. Profile Depth Before Expansion

**Named first batch (done):** Kodak Portra 160/400/800, Ektachrome E100, Fujifilm Provia 100F / Velvia 50, Kodak T-Max 400, Ilford Delta 400, Fujifilm Acros 100 II — public-datasheet-inspired profiles with brand names and `source{}` attribution. Color spectral stocks are approximate character models (not licensed LUTs). New B&W stocks use single-curve + CI×time morph except where noted.

Still prefer **deepening** over a second large catalog expansion. Tri-X 400 remains the only stock with digitized multi-time characteristic **curve families** (Kodak F-4017 D-76 / T-MAX). Other film×chem pairs morph from a single base curve until datasheet curve families are digitized.

**Chemistries (v1 → v1.2):** Develop exposes named developers + tank minutes from public datasheets.

**Next critique focus:**
- Side-by-side Tri-X D-76 @ 6 / 8 / 12 min vs real tanks — does the family feel right?
- Digitize remaining Tri-X chemistries (HC-110, XTOL) when plots exist; Ilford stocks need multi-time plots or densitometer sessions.
- Do D-76 vs T-MAX curve families feel distinct (not just grain bias)?
- Do named Portra / Fuji slide stocks feel distinct enough, or do they need densitometer recalibration first?
- Which profile currently feels weakest?
- Would one carefully refined additional paper (e.g., a different surface) add more value than three mediocre new films?

**Explicit non-goal for now:** analog camera / lens catalogs at ingest. Ingest stays digital raw → linear DN; capture-character (format, vignette, flare, MTF) is a later layer once film/paper feel is strong.

---

## 4. Performance & Preview Quality Under Real Use

As you work with larger real raws:
- Does live preview stay responsive?
- Is the preview resolution high enough for critical judgment while dragging?
- Are there any visible artifacts or inconsistencies when switching rapidly between controls?

---

## 5. Path Toward a Real Product (Medium-term)

Once the process feel is stronger:
- What is the minimum set of features required before this could be shown to other serious film photographers for feedback?
- Gradio is fine for development. When does it become the limiting factor for a polished experience?
- What would a first “serious beta” look like (even if still local/desktop)?

---

*Use this document as the ordered backlog for critique cycles. Fill Calibration_Notes.md after side-by-side sessions; implement those notes before expanding the profile catalog.*
