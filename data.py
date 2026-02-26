"""
Load FinanceBench JSONL into example_id, context, query, gold_answer.
Same logic as compressor data.py; kept self-contained.
"""

import json
from pathlib import Path
from typing import List, TypedDict


class Example(TypedDict):
    example_id: str
    context: str
    query: str
    gold_answer: str


def load_financebench(jsonl_path: str) -> List[Example]:
    """Load JSONL and extract (example_id, context, query, gold_answer) per row."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"FinanceBench path not found: {jsonl_path}")
    examples: List[Example] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = row.get("financebench_id", "")
            query = row.get("question", "")
            gold_answer = row.get("answer", "")
            seen: set = set()
            parts: List[str] = []
            for ev in row.get("evidence", []) or []:
                key = (ev.get("evidence_doc_name"), ev.get("evidence_page_num"))
                if key in seen:
                    continue
                seen.add(key)
                text = ev.get("evidence_text_full_page", "")
                if text:
                    parts.append(text)
            context = "\n\n".join(parts)
            examples.append(
                Example(
                    example_id=example_id,
                    context=context,
                    query=query,
                    gold_answer=gold_answer,
                )
            )
    return examples
