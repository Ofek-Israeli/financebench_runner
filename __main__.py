"""
CLI: run FinanceBench examples through local LLM via SGLang.
  python -m financebench_runner --config .config --input path/to/financebench_open_source.jsonl --output results.json [--limit N]

With --start-server the runner starts an SGLang server automatically and
stops it when done (same behaviour as compressor_2's evolution loop).
"""

import argparse
import json
import logging
import sys

from .runner import run_financebench


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run FinanceBench examples on a local LLM (SGLang)"
    )
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--input", required=True, help="Path to FinanceBench JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSON file")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N examples (for testing)",
    )
    ap.add_argument(
        "--logit-bias",
        default=None,
        metavar="JSON_FILE",
        help='Path to a JSON file mapping token IDs (strings) to bias floats, e.g. {"123": -100, "456": 5.0}',
    )
    ap.add_argument(
        "--logit-processor",
        default=None,
        metavar="PY_FILE",
        help="Path to a Python file containing a LearnedBloatAxisProcessor class "
             "(CustomLogitProcessor subclass). Requires SGLang server started with "
             "--enable-custom-logit-processor.",
    )
    ap.add_argument(
        "--correctness",
        action="store_true",
        default=False,
        help="Run correctness evaluation on each example using an OpenAI judge "
             "(requires OPENAI_API_KEY). Adds is_correct, correctness_confidence, "
             "correctness_reasoning, correctness_category to each output object.",
    )
    ap.add_argument(
        "--correctness-model",
        default=None,
        metavar="MODEL",
        help="OpenAI model for correctness evaluation (default: from config or gpt-4o)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max tokens per LLM reply (overrides config file when set)",
    )

    # --- SGLang server management ---
    ap.add_argument(
        "--start-server",
        action="store_true",
        default=False,
        help="Start an SGLang server automatically before running and stop it "
             "when done. The model and port are taken from the config file.",
    )
    ap.add_argument(
        "--gpu-id",
        default=None,
        metavar="ID",
        help="Physical GPU index for the SGLang server (sets "
             "CUDA_VISIBLE_DEVICES for the server process). "
             "Only used with --start-server.",
    )
    ap.add_argument(
        "--sglang-extra-args",
        default="",
        metavar="ARGS",
        help="Extra CLI arguments forwarded to sglang.launch_server "
             "(e.g. '--mem-fraction-static 0.8'). Only used with --start-server.",
    )

    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logit_bias = None
    if args.logit_bias:
        with open(args.logit_bias) as f:
            logit_bias = json.load(f)
        if not isinstance(logit_bias, dict):
            logging.error("--logit-bias file must contain a JSON object")
            return 1
    try:
        run_financebench(
            config_path=args.config,
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
            logit_bias=logit_bias,
            logit_processor_path=args.logit_processor,
            run_correctness=args.correctness,
            correctness_model=args.correctness_model,
            max_new_tokens=args.max_new_tokens,
            start_server=args.start_server,
            gpu_id=args.gpu_id,
            sglang_extra_args=args.sglang_extra_args,
        )
        return 0
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
