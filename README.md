# Digital Negative + Virtual Darkroom

Desktop darkroom workflow for digital capture: ingest a camera raw as a **Digital Negative** (latent image), then develop and print with photographer-facing controls.

## What’s working now

1. Ingest raw/image → **linear scene-referred Digital Negative** (CIE XYZ for raws)
2. Film stocks: **HP5 Plus** (refined), **FP4 Plus**, **Delta 100**
3. Development: relative time / N±, grain, developer styles, mild seed-controlled variation
4. Print: exposure (stops), multigrade grade with **filter speed**, papers  
   **Multigrade Standard**, **Multigrade Warmtone**, **Fiber Glossy**
5. Ritual Gradio UI: **live** Develop/Print previews; Commit locks the stage
6. CLI

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

# Sequential UI
python scripts/run_darkroom_ui.py
# open http://127.0.0.1:7860
```

## Ingest design

Camera raws are demosaiced to **linear CIE XYZ** with camera white balance and **no display tone curve** (`gamma=(1,1)`). The Digital Negative stores that payload; luminance for B&W development is the **Y** channel.

Rendered JPEGs/PNGs use an inverse-sRGB approximation and are marked as such in metadata — prefer raws for a true latent-image path.

Color development (per-channel vs luma + color-difference) is deferred; the B&W foundation stays luminance-based.

## Project layout

```
profiles/films/          HP5, FP4, Delta 100, Tri-X 400
profiles/papers/         MG Standard, MG Warmtone
src/digital_negative/
  ingest.py              Raw/image → linear Digital Negative
  development.py         Log-E → density (+ grain)
  print_engine.py        Enlarger / paper stage
  pipeline.py            Orchestration
scripts/run_spike.py
scripts/run_darkroom_ui.py
docs/                    Starter + critique notes
```

## Tests

```bash
pytest -q
```

## Notes

Film curves are approximate digitizations of public manufacturer datasheets (Ilford / Kodak F-4017); source URLs live inside each profile JSON. Do **not** copy curves/code from GPLv3 / CC BY-SA research projects.
