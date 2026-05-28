#!/usr/bin/env bash
# Start the real vLLM inference host (Linux/WSL only — no native Windows wheels).
# On Windows, use the mock instead:  python -m nse.scripts.vllm_mock
set -euo pipefail

MODEL="${NSE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${NSE_VLLM_PORT:-8080}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --dtype auto
