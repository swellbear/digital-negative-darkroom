#!/usr/bin/env bash
# Download public sample camera raws from raw.pixls.us (CC0 archive).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${ROOT}/raws"
BASE="https://raw.pixls.us/download/data"
mkdir -p "$DEST"

download() {
  local rel="$1"
  local out="$2"
  local url="${BASE}/${rel}"
  echo "→ ${out}"
  curl -fL --retry 3 --retry-delay 2 -o "${DEST}/${out}" "$url"
}

download "Nikon/D40/DSC_1842.NEF" "nikon_d40_DSC_1842.NEF"
download "Nikon/D90/00001.NEF" "nikon_d90_00001.NEF"
download "Canon/EOS%2040D/_MG_0153.CR2" "canon_40d_MG_0153.CR2"
download "Canon/EOS%20550D/IMG_4047.CR2" "canon_550d_IMG_4047.CR2"
download "Sony/ILCE-6000/DSC01542.ARW" "sony_a6000_DSC01542.ARW"

echo
ls -lh "$DEST"
echo "Done. Files are in ${DEST}"
