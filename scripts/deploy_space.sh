#!/usr/bin/env bash
# Create or update the Hugging Face Space for this repo.
# Requires: hf auth login   OR   HF_TOKEN=hf_... in the environment
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v hf >/dev/null 2>&1 && ! python3 -c "import huggingface_hub" 2>/dev/null; then
  pip install -U "huggingface_hub[cli]"
fi

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

python3 - <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

root = Path(".").resolve()
token = os.environ.get("HF_TOKEN") or None
try:
    info = whoami(token=token)
except Exception as exc:
    print("Not logged in to Hugging Face.", file=sys.stderr)
    print("Run:  hf auth login", file=sys.stderr)
    print("Or:   export HF_TOKEN=hf_...", file=sys.stderr)
    print(f"Detail: {exc}", file=sys.stderr)
    sys.exit(1)

user = info.get("name") or info.get("fullname") or ""
if not user:
    print(f"Could not resolve username from whoami: {info!r}", file=sys.stderr)
    sys.exit(1)

space_id = os.environ.get("HF_SPACE_ID") or f"{user}/digital-negative-darkroom"
print(f"Deploying → https://huggingface.co/spaces/{space_id}")

api = HfApi(token=token)
create_repo(
    space_id,
    repo_type="space",
    space_sdk="gradio",
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(root),
    repo_id=space_id,
    repo_type="space",
    ignore_patterns=[
        ".venv/**",
        "output/**",
        "samples/raws/**",
        ".git/**",
        ".gradio/**",
        "**/__pycache__/**",
        ".pytest_cache/**",
        ".github/**",
        "**/*.pyc",
    ],
)
print(f"Done → https://huggingface.co/spaces/{space_id}")
PY
