"""
CLI: run FinanceBench examples through local LLM via SGLang.
  python -m financebench_runner --config .config --input path/to/financebench_open_source.jsonl --output results.json [--limit N]
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
        )
        return 0
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
