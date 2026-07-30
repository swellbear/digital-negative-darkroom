# Digital Negative + Virtual Darkroom

Desktop darkroom workflow for digital capture: ingest a camera raw as a **Digital Negative** (latent image), then develop and print with photographer-facing controls.

## What’s working now

1. Ingest raw/image → **linear scene-referred Digital Negative** (CIE XYZ for raws)
2. Film stocks: **HP5 Plus**, **FP4 Plus**, **Delta 100**, **Tri-X 400**
3. Development: named developers + minutes, N±, grain; Tri-X D-76 / T-MAX use multi-time curve families
4. Print: **base exposure in seconds** (enlarger timer), MG filtration + filter speed, papers
5. Dodge/burn: preset enlarger **cards** (oval/circle/finger/custom) waved over the live print; passes timed against the base clock with on-screen stop math
6. **First-print guide** in the UI (Ingest → Develop → easel ready in one click)
7. Ritual Gradio UI · CLI · Hugging Face Spaces entrypoint (`app.py`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: public sample raws
bash samples/fetch_raws.sh

# CLI
python scripts/run_spike.py
python scripts/run_spike.py samples/raws/nikon_d40_DSC_1842.NEF --film delta-100-v1 --paper mg-warmtone

# UI (local)
python scripts/run_darkroom_ui.py
# or: python app.py
# open http://127.0.0.1:7860
```

### First print (in the UI)

1. Open **First print (≈2 min)** and click **Run first-print guide** (or Commit Ingest → Commit Develop yourself).
2. Set **Base exposure (seconds)** — default 8s ≈ calibrated normal.
3. Pick a **card shape** (soft oval is fine).
4. Click **Start — wave over print →** and move over the **highlighted print on the right**.
5. When the timer ends, inspect the change. **Reset local work** clears dodge/burn only.

### Hugging Face Space (stable public demo)

Create a Gradio Space from this repo (SDK: Gradio, app file: `app.py`). Or from a clone:

```bash
pip install huggingface_hub
# requires `huggingface-cli login`
gradio deploy
```

`app.py` imports `build_ui()` so Spaces and local launches share the same UI.

## Ingest design

Camera raws are demosaiced to **linear CIE XYZ** with camera white balance and **no display tone curve** (`gamma=(1,1)`). The Digital Negative stores that payload; luminance for B&W development is the **Y** channel.

Rendered JPEGs/PNGs/HEIFs use an inverse-sRGB approximation and are marked as such in metadata — prefer raws for a true latent-image path. HEIF/HEIC/AVIF require `pillow-heif`.

Color development (per-channel vs luma + color-difference) is deferred; the B&W foundation stays luminance-based.

## Project layout

```
app.py                   HF Spaces / Gradio entrypoint
profiles/films/          HP5, FP4, Delta 100, Tri-X 400
profiles/papers/         MG Standard, MG Warmtone, Fiber Glossy
src/digital_negative/
  ingest.py              Raw/image → linear Digital Negative
  development.py         Log-E → density (+ grain)
  chemistry.py           Named developers + minutes
  dodge_burn.py          Card stamps + local light-time
  print_engine.py        Enlarger / paper stage
  pipeline.py            Orchestration
scripts/run_spike.py
scripts/run_darkroom_ui.py
docs/                    Starter + critique notes
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Notes

Film curves are approximate digitizations of public manufacturer datasheets (Ilford / Kodak F-4017). Tri-X **D-76** and **T-MAX** include multi-time curve families; **HC-110** and **XTOL** use datasheet CI×time morphs until multi-time D–logE plots are digitized. Source URLs live inside each profile JSON. Do **not** copy curves/code from GPLv3 / CC BY-SA research projects.
