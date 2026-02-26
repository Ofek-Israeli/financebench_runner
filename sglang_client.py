"""
SGLang client for local LLM generation (single-prompt completions).
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SGLangClient:
    """Thin client for local SGLang server."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout_s: float,
        max_retries: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if self.base_url.endswith("/v1"):
            self._url = f"{self.base_url}/completions"
        else:
            self._url = f"{self.base_url}/v1/completions"

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        logit_bias: Optional[dict[str, float]] = None,
    ) -> str:
        """Send one prompt to the SGLang server; return generated text only."""
        payload: dict = {
            "model": self.model_id,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "seed": seed,
        }
        if logit_bias:
            payload["logit_bias"] = logit_bias
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self._url,
                    json=payload,
                    timeout=self.timeout_s,
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                text = choices[0].get("text", "")
                if isinstance(text, str):
                    return text.strip()
                return str(text).strip()
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning("SGLang request attempt %s failed: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"SGLang generate failed after {self.max_retries} retries") from last_exc
