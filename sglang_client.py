"""
SGLang client for local LLM generation (single-prompt completions).

Supports optional external CustomLogitProcessor classes for constrained
decoding (e.g., reducing verbosity via learned bloat-axis penalties).
The processor is serialized via .to_str() and sent to the SGLang server
alongside custom_params.  See learning_grammar/output_processor.py for
how these processor files are generated.
"""

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class SGLangClient:
    """Thin client for local SGLang server with optional logit processor."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout_s: float,
        max_retries: int,
        logit_processor_class: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.logit_processor_class = logit_processor_class
        if self.base_url.endswith("/v1"):
            self._url = f"{self.base_url}/completions"
        else:
            self._url = f"{self.base_url}/v1/completions"
        if logit_processor_class is not None:
            logger.info(
                "CustomLogitProcessor configured — server needs "
                "--enable-custom-logit-processor flag"
            )

    def check_reachable(self, timeout_s: float = 5.0) -> bool:
        """Return True if the server is reachable (any HTTP response), False on connection/timeout."""
        try:
            requests.get(self.base_url, timeout=timeout_s)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

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
        if self.logit_processor_class is not None:
            payload["custom_logit_processor"] = self.logit_processor_class().to_str()
            if hasattr(self.logit_processor_class, "get_default_params"):
                payload["custom_params"] = self.logit_processor_class.get_default_params()
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
