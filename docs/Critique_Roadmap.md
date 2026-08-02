# Critique Roadmap — Ordered Backlog

**Updated:** 2026-07-31  
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Compass:** [`Digital_Negative_Darkroom_Critique_v5.md`](Digital_Negative_Darkroom_Critique_v5.md)

The architecture works. Remaining risk is **clarity and authenticity under feature weight**. Work in this order; do not skip ahead to expansion or packaging.

---

## P0 — Stabilize & Clarify the Core Experience (current)

Make the live Develop → Print loop coherent and hard to misread.

1. Stage / lock / chemistry path visually unambiguous (Live exploring vs Committed; B&W / C-41 / E-6).
2. Harden B&W ↔ Color and camera-roll frame switching (no stale controls, no leakage, no stuck locks).
3. Quarantine dodge/burn behind **Advanced** so the default Print path is paper → exposure → filtration → Commit.

**Explicit non-goals until P0 closes:** new films/papers, `.exe`, new major subsystems, camera/lens catalogs.

---

## P1 — Process Feel Calibration (highest product value after P0)

B&W Develop → Print authenticity first; color stays early/experimental.

1. Side-by-side real raws vs known film / darkroom (or scan proxies).
2. Log 2–3 findings per session in [`Calibration_Notes.md`](Calibration_Notes.md).
3. Implement only the highest-leverage curve / N± / developer / MG exposure changes.
4. Prefer deepening Tri-X / Ilford families over new stocks.

---

## P2 — Cohesion Over Expansion

- Deepen existing profiles and papers; no second large catalog push.
- Resist new major tools until the core loop feels excellent to a darkroom photographer.
- Keep `src/digital_negative/` independent of Gradio so a future desktop UI remains possible.
- Legal posture: public datasheet digitizations only; never import GPLv3/CC-BY-SA research profiles.

---

## P3 — Product Readiness (later)

- Minimum “serious beta” for other film photographers.
- Performance on large real raws.
- Clearer onboarding / first-print path.
- Desktop launcher / non-Gradio UI only after the above.

---

*Fill Calibration_Notes.md after side-by-side sessions; implement those notes before expanding the profile catalog.*
