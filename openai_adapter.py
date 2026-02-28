"""
Thin adapter wrapping the openai SDK to match the .chat(messages) interface
expected by minions_channel's RemoteVerdictEvaluator.

This avoids importing minions.clients (which transitively pulls in ollama).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, "/workspace/minions_channel")
from minions.usage import Usage


class OpenAIAdapter:
    """Minimal client matching the interface RemoteVerdictEvaluator expects."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set "
                "(required for correctness evaluation)"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[str], Usage]:
        """Call OpenAI chat completions; return (responses, usage)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return [text], usage
