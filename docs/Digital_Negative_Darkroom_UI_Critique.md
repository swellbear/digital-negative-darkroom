# Digital Negative Darkroom — UI-Specific Critique
**Repository:** https://github.com/swellbear/digital-negative-darkroom  
**Date:** 2026-07-31  
**Focus:** User interface structure, interaction design, clarity, and friction

---

## 1. Overall UI Character

The current interface is a custom Gradio application that has evolved far beyond a simple sequential form. It now uses:

- A narrow **icon rail** (Upload / Roll / Develop / Print / Frame / Log / New)
- A **collapsible drawer** for stage-specific controls
- A large **live preview** area with filmstrip and status banner
- A secondary **module panel** (Inspect, Curves, Recipes, Dodge & Burn, Crop)
- Heavy custom CSS + JavaScript for drawers, roll management, dodge/burn waving, zoom/pan, and status

**Intent is clear:** recreate a darkroom-like sequential experience with live feedback.  
**Reality:** the UI has become a dense, stateful single-page application with significant custom behavior layered on Gradio.

---

## 2. What Works Well

### Strengths
- **Live preview** of a theoretical print while adjusting Develop and Print controls is a major usability win.
- **Stage rail + drawer** model keeps the main preview large and gives a physical sense of moving between darkroom phases.
- **Photographer language** (base exposure in seconds, N±, filter speed, named developers, card shapes) is the right vocabulary.
- **First-print guide** lowers the barrier to getting a first successful result.
- **Camera roll** supports multi-image sessions and per-frame settings (valuable for real use).
- Dark theme and copper/accent styling attempt a coherent workshop aesthetic.
- Commit / Unlock model preserves the sequential ritual while still allowing revision.

These are real achievements. The UI is no longer a prototype form; it is an instrument.

---

## 3. Primary UI Problems

### 3.1 Cognitive Load Is High
A user must simultaneously track:

- Current rail / drawer
- Stage lock status (Upload → Develop → Print)
- Chemistry mode (B&W vs Color)
- Which frame is active in the camera roll
- Whether the large preview is live/theoretical or committed
- Optional tools (Dodge & Burn, Crop, Inspect, Curves)

This is more mental overhead than a real darkroom usually imposes. The original “simple sequential ritual” feeling is diluted.

### 3.2 Narrow Drawer Crowding
The control drawer is very narrow (~188px). Even with careful CSS:

- Labels and values can feel cramped
- Long film/paper names or helper text risk truncation
- Dense slider groups become harder to scan and operate precisely

A photographer making fine N± or exposure decisions benefits from more breathing room and clearer value readouts.

### 3.3 Live vs Locked Ambiguity
The large viewer shows a **theoretical final print**. This is powerful, but the distinction between:

- “I am still exploring”
- “This stage is locked into the Digital Negative”

is not always obvious enough. Users can lose track of what has actually been committed versus what is only a live preview.

### 3.4 Dodge & Burn Interaction
The current “paint card → start timer → wave pointer over print” model is ambitious and analog-inspired, but in practice it is one of the least intuitive and most fragile parts of the UI:

- Multi-step setup
- Hidden timer / JS synchronization
- Pointer must stay over the print for exposure to accumulate
- Visual feedback can be subtle
- Easy for the interaction to feel broken or magical rather than controllable

This feature carries disproportionate complexity relative to its current reliability and clarity.

### 3.5 Custom JS / Gradio Fragility
The UI relies heavily on:

- Hidden Gradio inputs driven by JavaScript
- Polling and MutationObservers
- Manual event dispatch for roll switch/remove
- Complex layout fitting logic

This delivers advanced behavior but creates a higher risk of subtle breakage, race conditions, and difficult debugging. It also makes the UI harder for a solo developer to keep stable as features continue to grow.

### 3.6 Camera Roll + Stage State Complexity
Per-frame settings, save/discard prompts on switch, and re-enabling controls after frame changes are necessary for a multi-image workflow, but they add another layer of state the user (and the code) must manage correctly. Recent commits show this area has required repeated fixes.

---

## 4. Secondary Issues

- Multiple preview resolutions (drag vs high) can produce visible quality jumps.
- Accordion / module panel on the right can hide important tools.
- Status banner and decision log are useful but compete for attention with the main image.
- Accessibility (keyboard, screen readers) appears secondary; many interactions are pointer/hover driven.
- Mobile / small-viewport behavior is constrained by the fixed rail + drawer + preview design.

---

## 5. Design Principles to Protect

When improving the UI, protect these:

1. **Large, high-quality live preview** remains the hero.
2. **Sequential ritual** (Upload → Develop → Print) should stay legible at a glance.
3. **Photographer language** over generic editing terms.
4. **Live feedback inside a stage**; Commit only locks the stage.
5. **Engine independence** — keep processing logic out of the Gradio/JS layer so a future native UI is possible.

---

## 6. Recommended UI Priorities (Ordered)

### P0 — Clarity & Trust
1. Make current stage + locked vs live state visually unmistakable (stronger banner, clearer lock indicators, explicit “Working preview” vs “Committed” labeling).
2. Harden chemistry mode switching and camera-roll frame switching so controls and previews never feel stale or broken.
3. Give the control drawer more usable width or progressive disclosure so primary Develop/Print controls are easy to read and adjust.

### P1 — Reduce Cognitive Load
1. Simplify or quarantine interactive Dodge & Burn (move to Advanced, or replace with a simpler local adjustment tool until the core loop is excellent).
2. Reduce the number of simultaneous competing panels when possible.
3. Make the first-print / default path even more guided and forgiving.

### P2 — Stability of the Custom Layer
1. Reduce reliance on hidden inputs + polling where Gradio can express the behavior more directly.
2. Add targeted UI regression tests for rail/drawer switching, roll frame switch, and mode visibility.
3. Document the intended mental model of the UI in one short “How this darkroom works” note visible to the user.

### P3 — Later Polish
- Performance with larger raws and longer rolls
- Keyboard access for core actions
- Cleaner inspect / zoom experience
- Eventual move toward a more controlled desktop UI if Gradio remains limiting

---

## 7. Summary Judgment

The UI has successfully moved from a basic sequential form into a real interactive instrument. Live preview, stage locks, photographer language, and the camera roll are meaningful strengths.

The current weaknesses are **density, ambiguity of live-vs-locked state, and the complexity of secondary tools (especially dodge/burn and the custom JS layer)**. These make the experience feel less intuitive and less reliable than the underlying process engine deserves.

**Highest-leverage UI work right now:**  
Make the core Upload → Develop → Print loop with live preview feel simple, obvious, and trustworthy. Everything else should be subordinate to that goal.

---

*End of UI Critique*
