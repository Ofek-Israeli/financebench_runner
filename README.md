# financebench_runner

Run [FinanceBench](https://github.com/patronus-ai/financebench) examples on a **local LLM** via **SGLang**. Output is a JSON file with `llm_answer` and `ground_truth_answer` per example.

## Install

From this directory (or with the repo on `PYTHONPATH`):

```bash
pip install -r requirements.txt
```

## Config (YAML)

Create a YAML config file (e.g. `.config`) with at least:

| Key | Type | Example |
|-----|------|---------|
| `model_id` | str | `meta-llama/Llama-3.1-8B-Instruct` |
| `temperature` | float | `0.7` |
| `sglang.base_url` | str | `http://localhost:30000` |
| `sglang.timeout_s` | float | `120.0` |
| `sglang.max_retries` | int | `3` |
| `max_new_tokens` | int | `512` |
| `top_p` | float | `0.95` |
| `seed` | int | `42` |
| `prompt_template` | str | See below |
| `example_indices` | list of int (optional) | `[0, 1, 5, 10]` — run only these 0-based indices; omit to run all |

**prompt_template** must contain `{context}` and `{query}`. Example:

```yaml
prompt_template: |
  Context:
  {context}

  Question:
  {query}

  Answer:
```

Optional **example_indices**: list of 0-based indices to run. If present, only those examples are run; if omitted, all examples are run (subject to `--limit`). Example: `example_indices: [0, 1, 2, 10, 20]`.

## Usage

```bash
# From repo root (parent of financebench_runner), so the package is on PYTHONPATH:
cd /path/to/Thesis/repos
PYTHONPATH=. python -m financebench_runner \
  --config financebench_runner/.config \
  --input financebench/data/financebench_open_source.jsonl \
  --output financebench_runner/results.json

# Optional: run only first N examples
PYTHONPATH=. python -m financebench_runner --config .config --input ../financebench/data/financebench_open_source.jsonl --output out.json --limit 5
```

## Output

Single JSON file: an **array of objects**, one per example:

```json
[
  {
    "example_id": "financebench_id_03029",
    "question": "What is the FY2018 capital expenditure amount...",
    "ground_truth_answer": "$1577.00",
    "llm_answer": "..."
  },
  ...
]
```

## Prerequisites

- A local LLM server that exposes an **OpenAI-compatible completions** endpoint (`/v1/completions` with `model`, `prompt`, `temperature`, `top_p`, `max_tokens`).
- FinanceBench JSONL (e.g. `financebench/data/financebench_open_source.jsonl`).

### Running the server

**Linux (with NVIDIA GPU):** Use [SGLang](https://github.com/sgl-project/sglang). Install with `pip install "sglang[all]"` (requires vLLM, Linux only), then in a separate terminal:

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --port 30000
```

Set in config: `sglang.base_url: "http://localhost:30000"`, `model_id: "meta-llama/Llama-3.2-3B-Instruct"`.

**macOS (and Linux without GPU):** SGLang’s server depends on vLLM, which **only supports Linux**. On macOS use **[Ollama](https://ollama.com)** instead:

1. Install Ollama: https://ollama.com
2. In a terminal, pull and run a small Llama model (keeps the server running in the foreground):
   ```bash
   ollama run llama3.2:3b
   ```
   Or run the server in the background and use the model by name in your config.
3. In `.config` set:
   - `sglang.base_url: "http://localhost:11434/v1"`
   - `model_id: "llama3.2:3b"` (or the model name you used with `ollama run`)

Ollama exposes an OpenAI-compatible API on port 11434; financebench_runner will send requests to `http://localhost:11434/v1/completions`.
