#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODEL_DIR"

if [[ ! -x bin/rife-ncnn-vulkan ]]; then
    wget -q https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
    unzip -q rife-ncnn-vulkan-20221029-ubuntu.zip
    rm -rf bin
    mv rife-ncnn-vulkan-20221029-ubuntu bin
    rm -f rife-ncnn-vulkan-20221029-ubuntu.zip
fi

test -x bin/rife-ncnn-vulkan
