"""
Run FinanceBench examples through a local LLM via SGLang and write JSON output.
"""

import importlib.util
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config
from .data import Example, load_financebench
from .sglang_client import SGLangClient

LOG = logging.getLogger(__name__)


def load_logit_processor(path: str) -> Optional[Any]:
    """Dynamically load a CustomLogitProcessor subclass from a Python file.

    Looks for a class named ``LearnedBloatAxisProcessor`` in the module
    defined by *path*.  Returns ``None`` when the file is missing or the
    class is not found.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        LOG.warning("Logit processor file not found: %s", p)
        return None
    try:
        spec = importlib.util.spec_from_file_location("learned_logit_processor", p)
        if spec is None or spec.loader is None:
            LOG.warning("Failed to load spec for logit processor: %s", p)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "LearnedBloatAxisProcessor"):
            LOG.info("Loaded logit processor from %s", p)
            return module.LearnedBloatAxisProcessor
        LOG.warning("No LearnedBloatAxisProcessor class found in %s", p)
        return None
    except Exception:
        LOG.exception("Error loading logit processor from %s", p)
        return None


def run_financebench(
    config_path: str,
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    logit_processor_path: Optional[str] = None,
    run_correctness: bool = False,
    correctness_model: Optional[str] = None,
) -> None:
    """
    Load config and data, run each example through SGLang, write JSON.
    Each output object has example_id, question, ground_truth_answer, llm_answer.
    """
    cfg = load_config(config_path)
    sg = cfg.get("sglang", {})
    processor_cls = load_logit_processor(logit_processor_path) if logit_processor_path else None
    client = SGLangClient(
        base_url=str(sg.get("base_url", "")),
        model_id=str(cfg.get("model_id", "")),
        timeout_s=float(sg.get("timeout_s", 60)),
        max_retries=int(sg.get("max_retries", 3)),
        logit_processor_class=processor_cls,
    )
    if not client.check_reachable():
        LOG.error(
            "Cannot reach LLM server at %s. Is Ollama/SGLang running? Start it first, then re-run.",
            client.base_url,
        )
        sys.exit(1)
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
            LOG.warning(
                "example_indices out of range (0..%s, dataset has %s examples) skipped: %s. "
                "Edit .config or run 'make menuconfig' to fix.",
                n - 1,
                n,
                out_of_range[:20],
            )
    if limit is not None:
        examples = examples[:limit]
    template = str(cfg.get("prompt_template", ""))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    seed = int(cfg.get("seed", 42))

    concurrency = int(cfg.get("concurrency", 0))
    if concurrency < 1:
        concurrency = len(examples)

    def _run_one(idx_ex):
        i, ex = idx_ex
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
        return {
            "example_id": ex["example_id"],
            "question": ex["query"],
            "ground_truth_answer": ex["gold_answer"],
            "llm_answer": llm_answer,
        }

    results_map: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one, (i, ex)): i
            for i, ex in enumerate(examples)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results_map[idx] = fut.result()

    results: List[Dict[str, Any]] = [results_map[i] for i in range(len(examples))]

    if run_correctness:
        _evaluate_correctness(results, cfg, correctness_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Wrote %s (%s examples)", output_path, len(results))

    _print_summary(results, run_correctness)


def _evaluate_correctness(
    results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    correctness_model: Optional[str],
) -> None:
    """Run correctness evaluation on all results using RemoteVerdictEvaluator."""
    import sys
    if "/workspace/minions_channel" not in sys.path:
        sys.path.insert(0, "/workspace/minions_channel")
    from minions_channel.evaluate.correctness import RemoteVerdictEvaluator
    from .openai_adapter import OpenAIAdapter

    model = correctness_model or str(cfg.get("correctness_model", "gpt-4o"))
    tolerance = float(cfg.get("correctness_tolerance", 0.10))
    api_key_env = str(cfg.get("correctness_api_key_env", "OPENAI_API_KEY"))

    adapter = OpenAIAdapter(model=model, api_key_env=api_key_env)
    evaluator = RemoteVerdictEvaluator(
        remote_client=adapter, numerical_tolerance=tolerance,
    )

    LOG.info(
        "Running correctness evaluation (model=%s, tolerance=%.0f%%)...",
        model, tolerance * 100,
    )
    for i, r in enumerate(results):
        LOG.info(
            "Evaluating correctness %d/%d: %s",
            i + 1, len(results), r.get("example_id", ""),
        )
        ev = evaluator.evaluate(
            predicted=r.get("llm_answer", ""),
            ground_truth=r.get("ground_truth_answer", ""),
            question=r.get("question", ""),
        )
        r["is_correct"] = ev.is_correct
        r["correctness_confidence"] = ev.confidence
        r["correctness_reasoning"] = ev.reasoning
        r["correctness_category"] = ev.category


def _print_summary(results: List[Dict[str, Any]], has_correctness: bool) -> None:
    """Print a summary to stdout."""
    n = len(results)
    if n == 0:
        return
    mean_len = sum(len(r.get("llm_answer", "")) for r in results) / n
    print(f"\n=== Summary ===")
    print(f"Examples: {n}")
    print(f"Mean output length: {mean_len:.1f} chars")
    if has_correctness:
        num_correct = sum(1 for r in results if r.get("is_correct", False))
        ratio = num_correct / n
        print(f"Correctness: {num_correct}/{n} ({ratio * 100:.1f}%)")
