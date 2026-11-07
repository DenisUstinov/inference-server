#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

export HF_HOME="$MODEL_DIR/weights"

uv sync
uv run python -c "import torch; from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)"
uv run python -c "import torch, torchaudio, tempfile, pathlib; p = pathlib.Path(tempfile.mkdtemp()) / 't.wav'; torchaudio.save(str(p), torch.zeros(1, 24000), 24000); torchaudio.load(str(p))"
