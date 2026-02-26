"""
Run FinanceBench examples through a local LLM via SGLang and write JSON output.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config
from .data import Example, load_financebench
from .sglang_client import SGLangClient

LOG = logging.getLogger(__name__)


def run_financebench(
    config_path: str,
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    logit_bias: Optional[Dict[str, float]] = None,
) -> None:
    """
    Load config and data, run each example through SGLang, write JSON.
    Each output object has example_id, question, ground_truth_answer, llm_answer.
    """
    cfg = load_config(config_path)
    sg = cfg.get("sglang", {})
    client = SGLangClient(
        base_url=str(sg.get("base_url", "")),
        model_id=str(cfg.get("model_id", "")),
        timeout_s=float(sg.get("timeout_s", 60)),
        max_retries=int(sg.get("max_retries", 3)),
    )
    examples: List[Example] = load_financebench(input_path)
    # Optional: run only specific indices from config (0-based)
    raw_indices = cfg.get("example_indices")
    if raw_indices is not None:
        if not isinstance(raw_indices, list):
            raise ValueError("config example_indices must be a list of integers")
        indices = [int(i) for i in raw_indices]
        n = len(examples)
        examples = [examples[i] for i in indices if 0 <= i < n]
        out_of_range = [i for i in indices if i < 0 or i >= n]
        if out_of_range:
            LOG.warning("example_indices out of range (0..%s) skipped: %s", n - 1, out_of_range[:20])
    if limit is not None:
        examples = examples[:limit]
    template = str(cfg.get("prompt_template", ""))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    seed = int(cfg.get("seed", 42))

    results: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        LOG.info("Running example %s/%s: %s", i + 1, len(examples), ex["example_id"])
        prompt = template.format(context=ex["context"], query=ex["query"])
        llm_answer = client.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            logit_bias=logit_bias,
        )
        results.append({
            "example_id": ex["example_id"],
            "question": ex["query"],
            "ground_truth_answer": ex["gold_answer"],
            "llm_answer": llm_answer,
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Wrote %s (%s examples)", output_path, len(results))
