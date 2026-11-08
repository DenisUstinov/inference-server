#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

export HF_HOME="$MODEL_DIR/weights"

uv sync

uv run python -c "import torch; from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)"
uv run python -c "import torch, torchaudio, tempfile, pathlib; p = pathlib.Path(tempfile.mkdtemp()) / 't.wav'; torchaudio.save(str(p), torch.zeros(1, 24000), 24000); torchaudio.load(str(p))"
uv run python -c "import torch; from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-large-v3-turbo', dtype=torch.float32, device='cpu')"
uv run python - <<'PY'
import pathlib
import tempfile

import torch
import torchaudio
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt

model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)
tmp = pathlib.Path(tempfile.mkdtemp())
ref = tmp / 'ref.wav'
torchaudio.save(str(ref), torch.zeros(1, 48000), 24000)
prompt = model.create_voice_clone_prompt(str(ref), ref_text='reference transcript', preprocess_prompt=False)
prompt_path = tmp / 'prompt.pt'
prompt.save(str(prompt_path))
loaded = VoiceClonePrompt.load(str(prompt_path))
assert loaded.ref_text == prompt.ref_text
print('prompt smoke ok')
PY
