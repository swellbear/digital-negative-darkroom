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

Resist the urge to add many more films/papers yet.

**Next critique focus:**
- Take the existing three films and three papers and push their quality.
- Which profile currently feels weakest?
- Are there specific tonal regions (toe, shoulder, midtones) that consistently feel off?
- Would one carefully refined additional paper or film (e.g., a higher-contrast option or a different paper surface) add more value than three mediocre new ones?

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
