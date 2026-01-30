#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8002}"

# OpenAI-compatible server
python3 -m vllm.entrypoints.openai.api_server       --model "${MODEL_NAME}"       --host 0.0.0.0       --port "${PORT}"       --served-model-name "${MODEL_NAME}"
