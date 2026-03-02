# financebench_runner

Run [FinanceBench](https://github.com/patronus-ai/financebench) examples on a **local LLM** via **SGLang**. Output is a JSON file with `llm_answer` and `ground_truth_answer` per example.

## Install

From this directory (or with the repo on `PYTHONPATH`):

```bash
pip install -r requirements.txt
```

## Configuration

Configure the runner the same way as **compressor_2**: run **`make menuconfig`** from this directory for the interactive ncurses menu. This writes a Kconfig-format `.config`. The runner accepts both:

- **Kconfig .config** (from `make menuconfig` or `make defconfig`) — same style as compressor_2.
- **YAML .config** — legacy format; if your `.config` is YAML it is still loaded. The first time you run `make menuconfig` with a YAML `.config`, it is backed up to `.config.yaml.bak`.

```bash
make menuconfig   # Interactive configuration (ncurses)
make defconfig    # Load default configuration
```

## Config (YAML or Kconfig)

Create or edit a config file (e.g. `.config`). If using YAML, it must contain at least:

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
| `correctness_model` | str (optional) | `gpt-4o` — OpenAI model for correctness evaluation |
| `correctness_tolerance` | float (optional) | `0.10` — numerical tolerance for correctness (10%) |

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

# With a custom logit processor (requires SGLang server with --enable-custom-logit-processor):
PYTHONPATH=. python -m financebench_runner \
  --config financebench_runner/.config \
  --input financebench/data/financebench_open_source.jsonl \
  --output financebench_runner/results.json \
  --logit-processor path/to/learned_logit_processor.py

# With correctness evaluation (requires OPENAI_API_KEY):
PYTHONPATH=. python -m financebench_runner \
  --config financebench_runner/.config \
  --input financebench/data/financebench_open_source.jsonl \
  --output financebench_runner/results.json \
  --correctness

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

With `--correctness`, each object also includes:

```json
{
  "is_correct": true,
  "correctness_confidence": 0.95,
  "correctness_reasoning": "...",
  "correctness_category": "numerical"
}
```

A summary is printed at the end:

```
=== Summary ===
Examples: 34
Mean output length: 245.3 chars
Correctness: 22/34 (64.7%)
```

## Custom logit processor

The `--logit-processor` flag accepts a path to a Python file that defines a `LearnedBloatAxisProcessor` class (a `CustomLogitProcessor` subclass from SGLang). This is the same format produced by `learning_grammar/output_processor.py`. The class must provide:

- `.to_str()` -- serializes the processor so it can be sent to the SGLang server.
- `.get_default_params()` (classmethod, optional) -- returns a dict of default `custom_params`.

The SGLang server must be started with the `--enable-custom-logit-processor` flag for this to work. Ollama does not support custom logit processors.

## Prerequisites

- A local LLM server that exposes an **OpenAI-compatible completions** endpoint (`/v1/completions` with `model`, `prompt`, `temperature`, `top_p`, `max_tokens`).
- FinanceBench JSONL (e.g. `financebench/data/financebench_open_source.jsonl`).

### Running the server

**Linux (with NVIDIA GPU):** Use [SGLang](https://github.com/sgl-project/sglang). Install with `pip install "sglang[all]"` (requires vLLM, Linux only), then in a separate terminal:

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --port 30000 --enable-custom-logit-processor
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
