#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUTPUT=""
MULTIPLIER=2
MODEL="rife-v4.6"
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --multiplier) MULTIPLIER="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --uhd) EXTRA_FLAGS+=("-u"); shift ;;
        --tta-spatial) EXTRA_FLAGS+=("-x"); shift ;;
        --tta-temporal) EXTRA_FLAGS+=("-z"); shift ;;
        *) exit 1 ;;
    esac
done

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$MODEL_DIR/bin/$MODEL"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "unknown rife model: $MODEL" >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/in" "$WORK/out" "$WORK/sorted"
ffmpeg -loglevel error -i "$INPUT" "$WORK/in/%08d.png"
FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")"
if [[ ! "$FPS" =~ ^[0-9]+/[0-9]+$ ]]; then
    exit 1
fi
FRAMES="$(ls "$WORK/in" | wc -l)"
"$MODEL_DIR/bin/rife-ncnn-vulkan" -i "$WORK/in" -o "$WORK/out" \
    -m "$MODEL_PATH" -n "$((FRAMES * MULTIPLIER))" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

index=0
for frame in $(ls "$WORK/out" | sort -V); do
    cp "$WORK/out/$frame" "$(printf '%s/%08d.png' "$WORK/sorted" "$index")"
    index=$((index + 1))
done

OUT_FPS="$(python3 -c "num, den = '$FPS'.split('/'); print(float(num) / float(den) * $MULTIPLIER)")"
ffmpeg -loglevel error -y -framerate "$OUT_FPS" -i "$WORK/sorted/%08d.png" -i "$INPUT" \
    -map 0:v -map 1:a? -c:v libx264 -pix_fmt yuv420p -c:a copy -shortest "$OUTPUT"
