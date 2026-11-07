#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=$(cd "$BASE_DIR" && python3 - <<'PY'
import tomllib
from pathlib import Path

manifest = tomllib.loads(Path("models.manifest.toml").read_text(encoding="utf-8"))
print(" ".join(m["name"] for m in manifest["models"]))
PY
)

for model in $MODELS; do
    bash "$BASE_DIR/models/$model/install.sh"
done
