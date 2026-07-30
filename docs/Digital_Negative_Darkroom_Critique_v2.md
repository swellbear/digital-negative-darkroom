# Digital Negative Darkroom — Critique v2
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Date:** 2026-07-29 (updated assessment)  
**Previous critique:** Addressed several key points; this is a fresh review of the current codebase.

---

## Overall Assessment

The project has moved from a solid technical spike to a coherent early product foundation.

**Major positive change since last review:**  
Ingest now produces a proper **linear scene-referred Digital Negative** (CIE XYZ for raws, `gamma=(1,1)`, camera white balance, no display tone curve). This was the highest-priority technical issue and has been fixed correctly. The README and code comments now explicitly protect the latent-image philosophy.

The sequential Gradio UI with explicit **Commit Ingest → Commit Develop → Commit Print** stages is also a meaningful step toward the ritual experience that differentiates this project.

Current state is no longer just “the idea works.” It is becoming a usable darkroom-style tool for black-and-white work.

---

## What’s Working Well

### Core Pipeline
- **Ingest**: Raws → linear CIE XYZ Digital Negative. Clear separation between scene-referred data and later creative stages. Non-raw images are handled with an inverse-sRGB approximation and correctly flagged in metadata.
- **Development**: Log-exposure mapping centered on scene mid-tone, characteristic curve application, relative time / contrast / developer style modifiers, density-domain grain with process seed.
- **Print**: Exposure in stops, multigrade grade, paper response curves, basic contrast nudge. Paper profiles exist for Multigrade Standard and Warmtone.
- **Digital Negative object**: Carries image payload + structured metadata, history, and committed stages.

### Product / Experience
- Photographer-facing control language is consistent.
- Sequential commit flow in the UI begins to embody the ritual intent.
- CLI remains useful for testing and batch work.
- Profiles are documented as approximate digitizations of public Ilford datasheets (correct legal posture).

### Engineering
- Clean module separation (`ingest`, `development`, `print_engine`, `pipeline`, `curves`, `grain`, `papers`).
- Synthetic scene + sample raw fetch script support rapid iteration.
- Tests are present.

---

## Remaining Gaps vs. Original Vision

| Area                        | Original Goal                          | Current Status                          | Priority |
|----------------------------|----------------------------------------|-----------------------------------------|----------|
| Digital Negative integrity | Latent image from capture              | Strong (linear XYZ on raws)             | Low     |
| B&W process feel           | Realistic develop + print experience   | Good foundation, needs tuning           | High    |
| Color                      | First-class                            | Explicitly deferred                     | Medium (later) |
| Ritual / sequential UX     | Deliberate stages + uncertainty        | Commit stages exist; still early        | High    |
| Film / paper library       | Useful set of stocks & papers          | 3 films (HP5, FP4, Delta 100), 2 papers | Medium  |
| Look fidelity              | Believable under real use              | Curve-based; needs your darkroom eye    | High    |
| Commercial product polish  | Desktop app people pay for             | Prototype (CLI + Gradio)                | Later   |

---

## Detailed Critique & Recommendations

### 1. Process Feel (Highest Remaining Priority)
The technical path is correct (log-E → density → print response). The next leap is making the *behavior* feel like real chemistry and an enlarger.

**Recommendations:**
- Spend time with real raws and side-by-side comparisons against actual film + darkroom prints.
- Tune the interaction of relative development, contrast modifier, and developer styles until push/pull and “High Energy” vs “High Definition” behave the way you expect from experience.
- In the print stage, refine how grade and exposure interact so that changing filtration feels familiar to someone who has stood at a multigrade enlarger.
- Consider adding a simple dodge/burn tool once the global print response feels right.

Your hundreds of hours in a real darkroom are the most valuable calibration tool available. Use them.

### 2. Sequential Ritual UX
The Commit Ingest → Develop → Print flow is a good start.  

**Next steps:**
- Make the visual and interaction design reinforce commitment (e.g., clearer stage progression, optional “lock” after commit, visible history of decisions).
- Introduce mild, seed-controlled variability that is noticeable but not chaotic — enough to preserve some of the living quality of analog materials.
- Avoid turning the UI into a conventional slider-heavy editor. The sequential nature is a feature.

### 3. Profile Library
Three Ilford B&W films and two multigrade papers is a sensible minimal set.  

**Recommendation:**  
Deepen quality before expanding quantity. One more carefully tuned film or paper that responds beautifully to the controls is more valuable right now than five mediocre additions.

### 4. Color
Correctly deferred. Keep the B&W luminance path clean. When color is added, decide explicitly between per-channel development and a luma + color-difference model; do not let it pollute the current foundation.

### 5. Ingest Edge Cases
Linear XYZ path for raws is solid. Continue to treat rendered JPEGs/PNGs as second-class (approximate) inputs and keep the metadata honest about encoding.

### 6. Engineering Notes
- Grain NaN fix on real files was necessary and good.
- Keep processing logic independent of the Gradio UI so a more polished desktop front-end can replace it later without rewriting the engine.

---

## Suggested Focus Order (Solo Founder)

1. **Tune the B&W process feel** with real files and your darkroom judgment (development response + print grade/exposure behavior).
2. **Strengthen the sequential ritual** in the UI so the commit stages feel meaningful.
3. **Refine existing profiles** (and add at most one more film/paper) rather than rapidly expanding the library.
4. Only then consider color architecture or a more permanent UI framework.

---

## Summary

**Previous major gap (ingest linearization) has been closed correctly.**  

The project now has:
- A legitimate Digital Negative
- A working develop → print pipeline
- Photographer-oriented controls
- The beginning of a sequential darkroom experience

The remaining work is less about “does the architecture work?” and more about **making the process feel authentic** and turning the prototype into a tool that photographers who know the real darkroom will respect.

You are in a strong position. The combination of a clean technical foundation and your practical darkroom experience is the project’s real advantage — protect and exploit that.

---

*End of Critique v2*