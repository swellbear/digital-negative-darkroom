# Digital Negative + Virtual Darkroom

Desktop darkroom workflow for digital capture: ingest a camera raw as a **Digital Negative** (latent image), then develop and print with photographer-facing controls.

## What’s working now

1. Ingest raw/image (or synthetic test scene) → Digital Negative
2. Film stocks: **HP5 Plus**, **FP4 Plus**
3. Development: relative time, contrast, grain, developer style
4. Print: exposure (stops), multigrade grade, paper response
5. CLI + sequential Gradio UI

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CLI (synthetic scene if no file given)
python scripts/run_spike.py
python scripts/run_spike.py /path/to/file.NEF --film fp4-plus-v1 --print-grade 3

# Interactive UI
python scripts/run_darkroom_ui.py
# open http://127.0.0.1:7860
```

Outputs land in `output/`:

- `negatives/<uuid>.tiff` + `.json` — Digital Negative
- `*_developed.png` / `*_print.png` / `*_comparison.png`

## Project layout

```
profiles/films/          Characteristic curves (HP5, FP4)
profiles/papers/         Multigrade paper response
src/digital_negative/
  ingest.py              Raw/image → Digital Negative
  development.py         Log-E → density (+ grain)
  print_engine.py        Enlarger / paper stage
  pipeline.py            Orchestration
scripts/run_spike.py     CLI
scripts/run_darkroom_ui.py
docs/                    Product starter document
```

## Controls (photographer language)

| Stage | Control | Meaning |
|-------|---------|---------|
| Develop | Film stock | Characteristic curve + grain baseline |
| Develop | Relative development | 1.0 normal; >1 push; <1 pull |
| Develop | Contrast | Straight-line slope |
| Develop | Grain strength | Seeded micro-variation |
| Develop | Developer style | Standard / High Definition / High Energy |
| Print | Exposure | Stops of enlarger light |
| Print | Multigrade filtration | Grade 0–5 |
| Print | Print contrast | Fine nudge around selected grade |

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Notes

Film curves are approximate digitizations of public Ilford datasheets; source URLs live inside each profile JSON. Do **not** copy curves/code from GPLv3 / CC BY-SA research projects.
