---
title: Digital Negative Darkroom
emoji: 🎞️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.21.0
app_file: app.py
pinned: false
license: mit
short_description: Camera raw → film develop → enlarger print with dodge/burn
---

# Digital Negative + Virtual Darkroom

Desktop darkroom workflow for digital capture: ingest a camera raw as a **Digital Negative** (latent image), then develop and print with photographer-facing controls.

**Live Space:** after deploy → `https://huggingface.co/spaces/<you>/digital-negative-darkroom`

## What’s working now

1. Ingest raw/image → **linear scene-referred Digital Negative** (CIE XYZ for raws)
2. Film stocks: **HP5 Plus**, **FP4 Plus**, **Delta 100**, **Tri-X 400**
3. Development: named developers + minutes, N±, grain; Tri-X D-76 / T-MAX use multi-time curve families
4. Print: **base exposure in seconds** (enlarger timer), MG filtration + filter speed, papers
5. Dodge/burn: preset enlarger **cards** waved over the live print; **scroll** to resize; burn darkens / dodge lightens
6. **First-print guide** — one click to easel-ready
7. Ritual Gradio UI · CLI · Hugging Face Spaces (`app.py`)

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: public sample raws
bash samples/fetch_raws.sh

python app.py
# open http://127.0.0.1:7860
```

### First print (in the UI)

1. Click **Run first-print guide** (or Commit Ingest → Commit Develop).
2. Set **Base exposure (seconds)** — default 8s ≈ calibrated normal.
3. Pick a **card shape** (soft oval is fine). **Scroll** over the print to resize.
4. **Start — wave over print →** and move over the highlighted print.
5. When the timer ends, compare the pass. **Reset local work** clears dodge/burn only.

## Deploy to Hugging Face Spaces (stable URL)

### One-shot from your machine

```bash
hf auth login
bash scripts/deploy_space.sh   # creates/updates spaces/<you>/digital-negative-darkroom
```

Or: [New Space](https://huggingface.co/new-space) → SDK **Gradio**, app file **`app.py`**, link this GitHub repo.

### GitHub Action

Add repo secret `HF_TOKEN` (write access). Push to `main` runs `.github/workflows/deploy-space.yml`.

`app.py` fetches sample raws on first boot when `samples/raws/` is empty (raws are gitignored).

## Ingest design

Camera raws are demosaiced to **linear CIE XYZ** with camera white balance and **no display tone curve** (`gamma=(1,1)`). The Digital Negative stores that payload; luminance for B&W development is the **Y** channel.

Rendered JPEGs/PNGs/HEIFs use an inverse-sRGB approximation — prefer raws for a true latent-image path. HEIF/HEIC/AVIF require `pillow-heif`.

## Project layout

```
app.py                   HF Spaces / Gradio entrypoint
profiles/films/          HP5, FP4, Delta 100, Tri-X 400
profiles/papers/         MG Standard, MG Warmtone, Fiber Glossy
src/digital_negative/    ingest · develop · print · dodge/burn
scripts/run_darkroom_ui.py
scripts/deploy_space.sh
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Notes

Film curves are approximate digitizations of public manufacturer datasheets (Ilford / Kodak F-4017). Source URLs live inside each profile JSON. Do **not** copy curves/code from GPLv3 / CC BY-SA research projects.
