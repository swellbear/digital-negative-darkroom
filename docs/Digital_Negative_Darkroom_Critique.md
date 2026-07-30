# Digital Negative Darkroom — Code Critique
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Date:** 2026-07-29  
**Purpose:** Focused critique for use inside Cursor

---

## Overall Assessment

The project is in a strong early state. It already implements the core idea we defined:

- Camera raw (or image) → **Digital Negative** (latent image + metadata)
- Characteristic-curve based development
- Photographer-facing controls (relative development, contrast, grain, developer style)
- Print stage with multigrade-style controls
- Modular pipeline (`ingest` → `development` → `print_engine` → `pipeline`)
- CLI + Gradio UI
- Clear documentation that curves come from public Ilford datasheets (no GPLv3/CC-BY-SA research code)

This is a real Level-3 process-driven foundation, not just a sketch.

---

## What’s Working Well

- **Architecture** closely matches the intended design. Separation of concerns is clean.
- **Development engine** correctly works in log-exposure → density space and applies grain after the curve.
- **Controls** use photographer language rather than purely technical parameters.
- **Digital Negative** carries metadata, history, and `committed_stages` — good support for sequential workflow.
- **Synthetic test scene** is practical for rapid iteration.
- Project layout, README, and notes about data provenance are clear and responsible.

---

## Priority Issues & Recommendations

### 1. Ingest / Linearization (Highest Priority)

**Current behavior** (`ingest.py`):  
Raws are processed with `rawpy` using `output_color=rawpy.ColorSpace.sRGB` (and related post-processing), then treated as linear.

**Problem:**  
This bakes an sRGB tone curve and color space into the Digital Negative too early. A true latent-image approach wants scene-referred (or at least camera-linear) data.

**Recommendation:**
- Prefer `gamma=(1, 1)`, `no_auto_bright=True`, and a linear output path.
- Keep white balance handling explicit and documented.
- Clearly mark the current path as a pragmatic approximation if a full linear path is deferred.
- Goal: the image payload of the Digital Negative should be as close to linear scene-referred data as practical.

This is the single most important technical improvement for the integrity of the “Digital Negative” concept.

### 2. Developed Positive Preview

The current transmittance → positive preview conversion is serviceable for an early spike but simplified.  

As a darkroom printer you will likely want:
- Clearer separation between the developed negative and the print stage
- More responsive print curve behavior when changing exposure and grade

**Recommendation:** Treat the current positive preview as temporary. Strengthen the print engine so that changes in exposure and multigrade filtration feel closer to standing at an enlarger.

### 3. Color Path

The current pipeline is correctly oriented toward luminance / black-and-white.  

When color is added later, decide early whether development operates:
- Per-channel on RGB, or
- On a luminance + color-difference model

Do not let color requirements complicate the B&W foundation right now.

### 4. Ritual / Sequential Experience

Metadata already supports `committed_stages` and history — good.  

The real product differentiation will come from making the *experience* of moving Develop → Print feel deliberate (and optionally somewhat irreversible). The Gradio UI currently exposes controls; the sequential ritual is still mostly future UX work.

### 5. Curve Quality

Curves are correctly described as approximate digitizations of public Ilford datasheets, with sources recorded. This is the right legal and practical approach.  

Refine them over time using your darkroom eye and side-by-side testing rather than chasing perfect numerical matches early.

---

## Suggested Priority Order for Next Work

1. **Fix / improve raw → linear conversion** so the Digital Negative is more scene-referred.
2. Test with real raw files and real film scans side-by-side (your darkroom experience is the best judge of “feel”).
3. Strengthen the print stage so grade and exposure changes feel more like real enlarger work.
4. Add one more film profile and one more paper response to start feeling the system’s range.
5. Begin shaping the UI around sequential ritual rather than exposing every control at once.

---

## Summary for Cursor

The foundation is solid and already implements the core Digital Negative + process pipeline.  

**Immediate focus:** Clean up the ingest linearization path.  
Everything else can be iterated from a position of strength.

The combination of competent process modeling + your real darkroom experience is the project’s main advantage. Keep protecting that.

---

*End of critique*