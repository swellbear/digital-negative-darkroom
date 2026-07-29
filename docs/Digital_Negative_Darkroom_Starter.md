# Digital Negative + Virtual Darkroom
## Project Starter Document for Cursor

**Created:** July 29, 2026  
**Purpose:** Complete reference so you can begin implementation in Cursor immediately.

---

## 1. Core Vision

Build a commercial desktop application that treats digital capture as the creation of a **Digital Negative** (a latent image). The software then gives the photographer the same nuanced control and sequential decision-making experience as working with real film, chemistry, and an enlarger in a darkroom.

**Goals:**
- Both black-and-white and color
- High-fidelity look **and** ritual / controlled uncertainty
- Photographer-facing language and workflow (not just technical parameters)
- Software-first (works with normal camera raws)
- Bootstrap / solo-founder friendly

You are a professional photographer with hundreds of hours of real darkroom experience. This practical knowledge is a major advantage for tuning controls and results so they *feel* authentic.

---

## 2. Key Differentiators

| Aspect                    | Typical film emulation tools      | This project                                      |
|---------------------------|-----------------------------------|---------------------------------------------------|
| Starting point            | Finished digital image or scan    | Digital Negative treated as latent image          |
| Process depth             | Mostly look / LUT / preset        | Full develop → print pipeline with real controls  |
| Development control       | Limited or hidden                 | Explicit, meaningful parameters                   |
| Print stage               | Often weak or absent              | First-class enlarger-style controls               |
| Ritual & uncertainty      | Rarely considered                 | Deliberate design goal                            |
| Product intent            | Plugin or preset pack             | Standalone darkroom application                   |

**Closest existing references:**
- FilmLab → strong process simulation for *scanned film*, focused on print stage
- spektrafilm / agx-emulsion → excellent open research on spectral / physical simulation (GPLv3 + CC BY-SA profiles — do **not** copy code or profiles directly)

Your project is different because of the Digital Negative concept, the ritual UX, photographer-oriented controls, and commercial product packaging.

---

## 3. MVP Scope

### In Scope (v1)
- Ingest standard camera raw files
- Immediately convert them into a Digital Negative
- Film stock selection (start with a small set)
- Development stage with meaningful parameters
- Print stage with enlarger-style controls
- Sequential workflow feel
- Controlled variability (process seed for grain / micro-variation)
- Non-destructive editing + export (16-bit TIFF / JPEG)
- Desktop (macOS + Windows)

### Explicitly Out of Scope for MVP
- Custom camera hardware or firmware
- Full spectral engine
- Large / perfect film library
- Mobile apps
- Advanced AI tools or heavy retouching
- Cloud features

### Success Criteria
A photographer with darkroom experience can open a raw, choose film + development approach, move into printing controls, and feel that the decisions map to analog thinking. Results should feel process-driven rather than like a preset was applied.

---

## 4. High-Level Architecture

```
[Camera Raw]
      ↓
 Ingest & Normalize
      ↓
┌─────────────────────────────┐
│      Digital Negative       │  ← core data + metadata
└─────────────────────────────┘
      ↓
 Film Profile Layer
 (characteristic curve, base behavior, grain model)
      ↓
 Development Engine
 (time / contrast / grain modifiers → density image)
      ↓
 Print Engine
 (exposure, filtration, paper response, dodge/burn)
      ↓
 Finishing / Output
```

**Design principles:**
- Digital Negative is the single source of truth
- Stages are modular
- Parameters use photographer language
- Support both guided sequential mode and later free-form mode

---

## 5. Digital Negative Schema (v1)

**Recommended storage (MVP):**  
16-bit (or 32-bit float) TIFF or OpenEXR for the image + JSON sidecar for metadata.

### Metadata Structure (JSON)

```json
{
  "digital_negative_version": "1.0",
  "uuid": "unique-id",
  "created": "ISO-8601",
  "modified": "ISO-8601",

  "source": {
    "original_filename": "",
    "camera_make": "",
    "camera_model": "",
    "iso": 0,
    "shutter_speed": "",
    "aperture": "",
    "datetime_original": "",
    "raw_hash": ""
  },

  "ingest": {
    "white_balance": {
      "method": "as_shot | custom | none",
      "temperature": 0,
      "tint": 0
    },
    "orientation": 1
  },

  "film_profile": {
    "id": "hp5-plus-v1",
    "name": "HP5 Plus",
    "type": "bw | color_negative",
    "version": "1.0",
    "iso": 400
  },

  "development": {
    "enabled": true,
    "developer_id": "standard",
    "developer_name": "Standard",
    "relative_time": 1.0,
    "contrast_modifier": 0.0,
    "grain_strength": 1.0,
    "notes": ""
  },

  "print": {
    "enabled": true,
    "paper_id": "mg-standard",
    "paper_name": "Multigrade Standard",
    "filtration": {
      "type": "multigrade | color | none",
      "grade": 2.5,
      "values": {}
    },
    "overall_exposure": 0.0,
    "contrast": 0.0,
    "dodge_burn": []
  },

  "process_seed": 123456789,

  "history": [],

  "ui_state": {
    "current_stage": "development",
    "committed_stages": ["ingest"]
  },

  "extensions": {}
}
```

Keep the image payload as linear (or near-linear) scene-referred data. All creative decisions live in the metadata.

---

## 6. Process Approach (Level 3)

**Target:** Strong, believable process simulation using public characteristic curves + parametric controls.  
Full spectral engine and “perfect” profiles are **not** required for MVP or early commercial versions.

### Development Stage – Recommended First Controls
Use language you already think in from the darkroom:

- **Film choice** (profile)
- **Relative Development** (1.0 = normal, >1 = push, <1 = pull)
- **Contrast** (affects slope of the straight-line section)
- **Grain Strength**
- Optional simple **Developer style** (Standard / High Definition / High Energy, etc.)

These modifiers reshape a base characteristic curve and control grain.

### Print Stage – Recommended First Controls
- Overall exposure
- Contrast / filtration (multigrade-style for B&W, basic color balance for color)
- Paper type (a few response curves)
- Basic local dodge & burn

### Data Strategy
- Digitize characteristic curves yourself from official public manufacturer datasheets (Ilford, Kodak, etc.).
- Start with a small high-quality set (e.g. HP5 Plus, FP4 Plus, Portra 400).
- Store curves as clean point lists that can be interpolated (spline).
- Document the exact source and digitization date inside each profile.
- Do **not** copy profiles or code from spektrafilm / agx-emulsion (GPLv3 + CC BY-SA).

---

## 7. First Technical Spike (Do This First in Cursor)

**Goal:**  
Open a raw file → create a basic Digital Negative → apply one characteristic curve → show the result.

**Steps:**
1. Set up a minimal project.
2. Use a raw-reading library to load a test raw and convert to linear RGB (or luminance for B&W test).
3. Create a Digital Negative object (image array + minimal metadata).
4. Load one digitized characteristic curve (start with HP5 Plus).
5. Convert linear values → log exposure → interpolate density from the curve.
6. Convert density back to a viewable image.
7. Display before/after or the processed result.

Once this loop works, add:
- Development modifiers (relative time, contrast)
- A second film profile
- Simple print exposure/contrast
- Minimal UI that exposes the controls

---

## 8. Suggested Tech Starting Point

Choose what you can move fastest in:

- **Language:** Python is excellent for rapid image pipeline experimentation (`rawpy`, NumPy, OpenCV / scikit-image).
- **UI (early):** Gradio, Dear PyGui, or a simple custom window. Replace later if needed.
- **Alternative:** TypeScript + a web or Electron UI if you prefer that ecosystem.

Keep the core processing logic separate from the UI so you can change either side later.

---

## 9. Important Decisions Already Made

- Software-first (no custom camera required for MVP)
- Digital Negative is created **on ingest** of normal raws
- Level-3 process simulation is the realistic and sufficient target for a solo founder
- Your real darkroom experience is the primary tool for making controls and results feel authentic
- Start small and high-quality (few excellent profiles > many mediocre ones)
- Ritual and sequential feel are deliberate product features

---

## 10. Immediate Next Actions

1. Create a new project/folder in Cursor.
2. Copy this document into the project (or keep it as reference).
3. Choose language + basic libraries.
4. Digitize one characteristic curve (HP5 Plus recommended).
5. Implement the first technical spike described above.
6. Once the spike works, expand controls and add a simple sequential UI.

---

## 11. Notes on Future Growth

Later (after a working core and real user feedback):
- More film profiles
- Richer development models
- Better print tools and papers
- Stronger controlled uncertainty / ritual UX
- Optional deeper spectral techniques
- Tethered capture support
- Possible future capture-side Digital Negative (camera partnership or companion app)

Do not let these future items block the first working version.

---

**This document is intentionally practical.**  
It contains enough architecture and decision-making for you to begin coding productively while leaving room for the refinements that will come from actually building and testing with your darkroom eye.

Good luck. Start with the spike.  
The rest will become clearer once you have a living pipeline.