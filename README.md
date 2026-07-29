# Digital Negative + Virtual Darkroom

Desktop darkroom workflow for digital capture: ingest a camera raw as a **Digital Negative** (latent image), then develop and print with photographer-facing controls.

This repo currently contains the **first technical spike**:

1. Ingest a raw/image (or a built-in synthetic scene)
2. Create a Digital Negative (linear image + JSON metadata)
3. Apply a digitized **HP5 Plus** characteristic curve
4. Export density / positive previews and a before/after strip

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Synthetic scene (no raw file required)
python scripts/run_spike.py

# Or pass a camera raw / image
python scripts/run_spike.py /path/to/file.NEF
```

Outputs land in `output/`:

- `negatives/<uuid>.tiff` + `.json` — Digital Negative payload
- `<uuid>_density.png` — developed density visualization
- `<uuid>_positive.png` — simple positive preview
- `<uuid>_comparison.png` — linear DN vs developed preview

## Project layout

```
profiles/films/     Film characteristic curves (JSON point lists)
src/digital_negative/
  ingest.py         Raw/image → Digital Negative
  digital_negative.py
  curves.py         Profile load + spline interpolation
  development.py    Log-E → density → transmittance
  display.py        Preview helpers
  pipeline.py       Spike orchestration
scripts/run_spike.py
docs/               Starter / product reference
```

## Film profiles

`profiles/films/hp5-plus-v1.json` is an approximate digitization of the public Ilford HP5 Plus characteristic curve (ILFOTEC HC 1+31, 6½ min @ 20°C). Source URL and notes are stored inside the profile. Re-digitize from a high-resolution datasheet plot before shipping production profiles.

Do **not** copy curves or code from GPLv3 / CC BY-SA research projects (e.g. spektrafilm / agx-emulsion).

## Tests

```bash
source .venv/bin/activate
pip install pytest
pytest -q
```

## Next

- Development modifiers UI (relative time, contrast, grain)
- Second film profile (FP4 Plus / Portra 400)
- Print stage (exposure, multigrade filtration, paper response)
- Sequential darkroom UI shell
