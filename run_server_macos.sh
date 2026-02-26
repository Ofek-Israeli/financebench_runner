#!/usr/bin/env bash
# On macOS, SGLang server cannot run (vLLM is Linux-only). Use Ollama instead.
# This script starts the Ollama server and pulls a 3B model so financebench_runner can connect.

set -e
if ! command -v ollama &>/dev/null; then
  echo "Ollama is not installed. Install from https://ollama.com"
  echo "Then run this script again, or run: ollama serve  and  ollama run llama3.2:3b"
  exit 1
fi
echo "Pulling llama3.2:3b (one-time download)..."
ollama pull llama3.2:3b
echo "Starting Ollama server in the background (API at http://localhost:11434)..."
ollama serve &
OLLAMA_PID=$!
sleep 2
echo "Ollama server PID: $OLLAMA_PID"
echo "Use .config with: sglang.base_url: \"http://localhost:11434/v1\", model_id: \"llama3.2:3b\""
echo "Run financebench_runner in another terminal. To stop the server: kill $OLLAMA_PID"
wait $OLLAMA_PID
