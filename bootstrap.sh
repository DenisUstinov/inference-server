#!/usr/bin/env bash
set -euo pipefail
umask 022

BASE_DIR="${BASE_DIR:-/opt/inference-server}"
SERVICE_NAME="inference-server"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="0.12.0"

if [[ $EUID -ne 0 ]]; then
    exit 1
fi

SERVICE_USER="${SUDO_USER:-}"
if [[ -z "$SERVICE_USER" || "$SERVICE_USER" == "root" ]]; then
    echo 'bootstrap must be run via sudo from a non-root user' >&2
    exit 1
fi

apt-get update -qq
apt-get install -y -qq git curl ca-certificates ffmpeg python3 unzip wget

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh
fi

if [[ "$SOURCE_DIR" != "$BASE_DIR" ]]; then
    mkdir -p "$BASE_DIR"
    cp -a "$SOURCE_DIR"/. "$BASE_DIR"/
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$BASE_DIR"
chmod -R go-w "$BASE_DIR"

sudo -u "$SERVICE_USER" bash -c "
    set -e
    cd '$BASE_DIR/rpc'
    uv sync
    mkdir -p data/in data/out
"

if [[ -n "${HF_TOKEN:-}" ]]; then
    sudo -u "$SERVICE_USER" HF_TOKEN="$HF_TOKEN" bash "$BASE_DIR/install-models.sh"
else
    sudo -u "$SERVICE_USER" bash "$BASE_DIR/install-models.sh"
fi
chown -R root:root "$BASE_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$BASE_DIR/rpc/data"
LORA_CACHE="$BASE_DIR/models/stable-audio/project/optimized/tflite/models/tflite/lora_cache"
mkdir -p "$LORA_CACHE"
chown "$SERVICE_USER:$SERVICE_USER" "$LORA_CACHE"
find "$BASE_DIR/models" -mindepth 2 -maxdepth 2 -type d -name weights -exec chown -R "$SERVICE_USER:$SERVICE_USER" {} +
chmod -R go-w "$BASE_DIR"

sed -e "s|__USER__|$SERVICE_USER|g" -e "s|__BASE_DIR__|$BASE_DIR|g" \
    "$BASE_DIR/deploy/inference-server.service.in" \
    > "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
