#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

python model_memory_http_server.py --host "$HOST" --port "$PORT"
