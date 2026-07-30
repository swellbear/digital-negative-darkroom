# Digital Negative Darkroom — Critique v4
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Date:** 2026-07-30  
**Previous critiques:** v1–v3 addressed; this reflects the current live state of the codebase.

---

## Overall Assessment

The project has reached a meaningful new stage.

**Major recent advances:**
- Linear scene-referred Digital Negative (CIE XYZ) remains solid.
- **Live preview** is now implemented for Develop and Print controls.
- Commit buttons correctly act as **stage locks** rather than the only way to see changes.
- Ritual structure (Ingest → Develop → Print) is preserved while interaction is fluid.
- Profile set has grown slightly and HP5 has been refined.
- Mild seed-controlled process variation is present.
- Additional paper profile (Fiber Glossy) added.

The tool has moved from “correct process demonstration” to an early but usable interactive darkroom instrument for black-and-white work.

---

## What’s Working Well

### Core Technical Pipeline
- Ingest produces linear CIE XYZ for raws with camera white balance and no display tone curve.
- Development works in log-exposure → density space with relative development, contrast (N±), grain, and developer styles.
- Print stage supports exposure in stops, multigrade filtration (with filter speed), and multiple paper responses.
- Digital Negative maintains metadata, history, committed/locked stages, and process seed.

### Interaction Model (Important Improvement)
- Controls in Develop and Print stages update the preview **in real time**.
- Commit actions lock the current stage and record the decision.
- This hybrid model successfully balances:
  - Fluid exploration (live feedback)
  - Deliberate progression (stage locks + history)

This directly addresses the earlier request for real-time slider response without abandoning the sequential ritual.

### Product Foundations
- Photographer-facing language is consistent.
- Profiles remain based on public Ilford datasheet digitizations with proper attribution notes.
- Modular architecture continues to support future growth.
- CLI + Gradio UI both functional.

---

## Remaining Gaps & Priorities

### 1. Process Feel & Calibration (Highest Remaining Priority)
Live controls are only as good as the underlying response curves and modifiers.

**Focus:**
- Continue side-by-side testing with real raws against actual film and darkroom prints.
- Refine how relative development, contrast, and developer styles interact so the results match your experienced expectations.
- Improve the relationship between multigrade grade, filter speed, exposure, and paper response until it feels natural at the enlarger.

Your darkroom experience remains the most valuable tuning instrument available.

### 2. Ritual & UX Polish
The live + commit model is correct. Next refinements:
- Visual clarity of current stage and locked decisions.
- Smooth handling of “unlock / revise” if a user wants to go back.
- Making the history panel more readable and useful as a decision record.
- Ensuring mild process variation is perceptible but never chaotic.

### 3. Profile Quality vs Quantity
Current set (HP5 refined, FP4, Delta 100 + three papers) is reasonable.  
Prefer deepening the response quality of existing profiles over rapid expansion.

### 4. Color
Still correctly deferred. Keep the luminance-based B&W foundation clean.

### 5. Performance & Preview Quality
As live preview is used more heavily:
- Consider resolution strategies (fast preview while dragging, higher quality on release or commit).
- Watch for responsiveness with larger raw files.

### 6. Path Toward a Real Product
Gradio is excellent for prototyping. Longer-term, a more polished desktop UI will be needed for commercial use. Keep the processing engine independent of the current front-end.

---

## Status Relative to Original Vision

| Goal                              | Current Status                          | Notes |
|-----------------------------------|-----------------------------------------|-------|
| Digital Negative (latent image)   | Strong                                  | Linear XYZ ingest achieved |
| Develop + Print process controls  | Strong foundation                       | Live + lock model in place |
| Photographer language             | Present                                 | Consistent |
| Ritual / sequential experience    | Implemented and improved                | Live feedback + stage locks |
| Real-time exploration             | Now present                             | Key recent win |
| Believable process feel           | Improving, needs continued tuning       | Highest remaining work |
| Color support                     | Deferred                                | Correct decision |
| Commercial product polish         | Still prototype                         | Expected at this stage |

---

## Summary

**The previous highest UX friction (no live preview) has been resolved.**  

The project now has:
- A legitimate Digital Negative
- A working, interactive develop → print pipeline
- A thoughtful hybrid of fluid control and sequential ritual
- A clean technical base for further refinement

The most valuable next work is no longer architectural. It is **calibrating the process response** so that what the photographer sees while moving the controls feels authentic to real film and darkroom practice.

You are past the “does this idea work?” phase and into the “make it feel right” phase. That is the correct place to be.

---

*End of Critique v4*
