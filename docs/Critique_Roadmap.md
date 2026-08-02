# Critique Roadmap — Ordered Backlog

**Updated:** 2026-07-31  
**Repository:** https://github.com/swellbear/digital-negative-darkroom  

**Compasses:**
- Product: [`Digital_Negative_Darkroom_Critique_v5.md`](Digital_Negative_Darkroom_Critique_v5.md)
- UI: [`Digital_Negative_Darkroom_UI_Critique.md`](Digital_Negative_Darkroom_UI_Critique.md)

The architecture works. Remaining risk is **clarity, density, and authenticity under feature weight**. Work in this order; do not skip ahead to expansion or packaging.

---

## UI P0 — Clarity & Trust (current: drawer slice)

Make the core Upload → Develop → Print loop readable and trustworthy.

**Done (PR #86):**
1. Stage / lock / chemistry path unambiguous (Live exploring vs Committed; B&W / C-41 / E-6).
2. Harden B&W ↔ Color and camera-roll frame switching.
3. Quarantine dodge/burn behind **Advanced**.

**Now:**
4. Widen the control drawer and use progressive disclosure so primary Develop/Print controls are easy to read and adjust.
5. Seed an in-UI “How this darkroom works” note (Upload drawer).

**Explicit non-goals until UI P0 closes:** JS architecture rewrite, replacing dodge/burn, mobile layout, new films/papers, `.exe`.

---

## UI P1 — Reduce Cognitive Load

1. Keep Advanced dodge/burn subordinate; simplify further only if the core loop is excellent.
2. Reduce simultaneous competing panels when possible.
3. Make the first-print / default path more guided and forgiving.

---

## Product P1 — Process Feel Calibration

Runs **after** UI trust, not instead of it. B&W Develop → Print authenticity first; color stays early/experimental.

1. Side-by-side real raws vs known film / darkroom (or scan proxies).
2. Log 2–3 findings per session in [`Calibration_Notes.md`](Calibration_Notes.md).
3. Implement only the highest-leverage curve / N± / developer / MG exposure changes.
4. Prefer deepening Tri-X / Ilford families over new stocks.

---

## UI P2 — Custom-Layer Stability

1. Reduce reliance on hidden Gradio inputs + polling where Gradio can express behavior directly.
2. Add targeted UI regression tests for rail/drawer, roll switch, and mode visibility.
3. Keep the mental-model note visible; expand only if users still misread the ritual.

---

## UI / Product P3 — Later Polish

- Performance with larger raws and longer rolls
- Keyboard access for core actions
- Cleaner inspect / zoom
- Serious beta / desktop launcher / non-Gradio UI only after the above
- Cohesion over catalog expansion; legal posture unchanged (public datasheets only)

---

## Parked (do not start)

**Scan-corpus / ML calibration model** — using public 35mm negative scans (or a trained appearance model) to bulk up Level-3 feel. Interesting as a later Level-3 appearance bench fed by **owned or properly licensed** paired captures + datasheets; **not** scraped web training. Tabled 2026-07-31 until Product P1 hand calibration is underway.

**Instant peel-apart (Type 55 etc.)** — classic Polaroid +/- datasheets are strong, but the UI ritual differs from integral 600/SX-70 (peel timing, clearing bath, usable negative). Integral Instant mode shipped first; peel-apart stays parked until that path is calibrated.

---

*Fill Calibration_Notes.md after side-by-side sessions; implement those notes before expanding the profile catalog.*
