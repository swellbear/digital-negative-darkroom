# Digital Negative Darkroom — Critique v3
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Date:** 2026-07-29 / 2026-07-30  
**Focus:** Current state after recent commits + user request for real-time control feedback

---

## Overall Assessment

The project has progressed well and now has a coherent black-and-white darkroom foundation:

- Proper **linear scene-referred Digital Negative** (CIE XYZ for raws)
- Working develop → print pipeline with photographer-facing controls
- Explicit sequential ritual UI with stage commits and locks
- Small but usable set of Ilford-based film and paper profiles
- History tracking and process seed support

The technical spine is solid. The main remaining gap between the current tool and a compelling daily-use experience is **interaction fluidity**.

---

## What’s Working Well

### Technical Foundation
- Ingest correctly prioritizes linear CIE XYZ with camera white balance and no display tone curve.
- Development operates in log-exposure → density space with relative time, contrast, grain, and developer style modifiers.
- Print stage supports exposure (stops), multigrade grade, and paper response.
- Digital Negative carries structured metadata, history, and locked/committed stages.
- Modular code structure remains clean.

### Ritual Structure
- Gradio UI enforces Ingest → Develop → Print progression.
- Commit actions lock stages and record decisions in history.
- This successfully expresses the sequential darkroom philosophy.

### Product Direction
- Control language stays photographer-oriented.
- Documentation correctly notes the use of public datasheet digitizations and avoids GPLv3/CC-BY-SA research code.

---

## Current Gap: Live Feedback

**User request (explicit):**  
Ability to adjust controls (sliders) and see the changes on the photo **in real time**, without pressing Commit for every small adjustment.

**Current behavior:**  
Develop and Print controls only update the preview when the corresponding **Commit** button is pressed. Slider movement alone produces no visual feedback.

This creates unnecessary friction during the exploratory phase of development and printing — the exact moments when a photographer most wants immediate visual response.

### Recommended Model (Hybrid)

Keep the high-level ritual while adding fluid interaction inside stages:

| Stage     | Behavior                                      | Commit / Lock Role                  |
|-----------|-----------------------------------------------|-------------------------------------|
| Ingest    | Still requires Commit (creates the DN)        | Creates & locks the latent image    |
| Develop   | **All controls update preview live**          | Locks development decisions         |
| Print     | **All controls update preview live**          | Finalizes the print                 |

This preserves sequential intentionality while removing the need to Commit just to see what a slider does.

---

## Priority Recommendations

### 1. Implement Live Preview (Highest UX Priority)
- Wire Develop-stage controls (film, developer, relative time, contrast, grain) so changes immediately refresh the negative/lightbox preview.
- Wire Print-stage controls (paper, exposure, grade, contrast nudge) so changes immediately refresh the print preview.
- Keep the Commit buttons, but change their meaning to “Lock this stage and proceed.”
- Optional: lower-resolution preview while dragging for responsiveness, higher quality on release or on Commit.

### 2. Continue Tuning Process Feel
Live controls only become powerful once the underlying response feels authentic. Continue side-by-side testing with real files and real darkroom knowledge so that:
- Relative development and contrast modifiers behave like push/pull and grade changes you expect
- Multigrade filtration + exposure interaction feels familiar

### 3. Ritual Polish
Once live preview exists:
- Make the stage progression visually clearer
- Ensure history remains a useful record of locked decisions
- Consider mild, seed-based variation that is visible but not disruptive

### 4. Profile Depth Before Breadth
Three films and two papers are sufficient for now. Prefer refining the existing profiles’ response over rapidly adding more stocks.

### 5. Color (Still Deferred)
Correct decision. Keep the luminance-based B&W path clean.

---

## Summary vs. Original Vision

| Aspect                        | Status                                      |
|-------------------------------|---------------------------------------------|
| Digital Negative concept      | Strong                                      |
| Process pipeline (dev + print)| Strong foundation                           |
| Photographer-facing controls  | Present                                     |
| Sequential ritual             | Implemented (currently strict)              |
| Real-time exploration         | Missing — highest current friction          |
| Color                         | Deferred                                    |
| Commercial polish             | Still prototype                             |

**Bottom line:**  
The architecture and philosophy are in good shape. The next high-leverage improvement is making the Develop and Print stages respond live to control changes while retaining Commit as a deliberate stage-lock action. This directly addresses the user’s stated need and moves the tool from “correct process demo” toward “usable darkroom instrument.”

---

*End of Critique v3*