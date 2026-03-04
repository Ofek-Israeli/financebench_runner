"""
SGLang server lifecycle: start, stop, health-check.

Standalone version for financebench_runner — no dependency on compressor_2.
Starts an SGLang server subprocess, optionally pinned to a specific GPU via
CUDA_VISIBLE_DEVICES, and waits until the /health endpoint responds.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Any, Dict, Optional

import requests

LOG = logging.getLogger(__name__)


class SGLangServer:
    """Manage an SGLang server subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        gpu_id: Optional[str] = None,
        extra_args: str = "",
        health_timeout: int = 300,
        health_interval: int = 5,
        enable_custom_logit_processor: bool = False,
    ):
        self._model_path = model_path
        self._port = port
        self._gpu_id = gpu_id
        self._extra_args = extra_args
        self._health_timeout = health_timeout
        self._health_interval = health_interval
        self._enable_custom_logit_processor = enable_custom_logit_processor
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        if self._proc is not None:
            LOG.warning("SGLang already running (pid %s); stopping first", self._proc.pid)
            self.stop()

        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self._model_path,
            "--port", str(self._port),
        ]
        if self._enable_custom_logit_processor:
            cmd.append("--enable-custom-logit-processor")
        if self._extra_args:
            cmd.extend(shlex.split(self._extra_args))

        env = dict(os.environ)
        if self._gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = self._gpu_id

        LOG.info(
            "Starting SGLang (model=%s, port=%d, gpu=%s): %s",
            self._model_path, self._port, self._gpu_id or "default", " ".join(cmd),
        )
        self._proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        LOG.info("SGLang started (pid %s); waiting for health ...", self._proc.pid)
        self._wait_healthy()

    def stop(self) -> None:
        if self._proc is None:
            return
        pid = self._proc.pid
        LOG.info("Stopping SGLang (pid %s) ...", pid)
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            LOG.warning("SGLang did not exit after SIGTERM; sending SIGKILL")
            self._proc.kill()
            self._proc.wait(timeout=10)
        LOG.info("SGLang stopped (pid %s)", pid)
        self._proc = None

    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self._health_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"SGLang exited with code {self._proc.returncode} during health wait"
                )
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    LOG.info("SGLang healthy")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(self._health_interval)
        raise RuntimeError(
            f"SGLang did not become healthy within {self._health_timeout}s"
        )

    @classmethod
    def from_runner_config(
        cls,
        cfg: Dict[str, Any],
        gpu_id: Optional[str] = None,
        extra_args: str = "",
        enable_custom_logit_processor: bool = False,
    ) -> "SGLangServer":
        """Build from the nested dict that financebench_runner's load_config returns."""
        sg = cfg.get("sglang", {})
        base_url = str(sg.get("base_url", "http://localhost:8000/v1"))
        port = 8000
        if ":" in base_url.split("//")[-1]:
            port_str = base_url.split("//")[-1].split(":")[1].split("/")[0]
            try:
                port = int(port_str)
            except ValueError:
                pass

        return cls(
            model_path=str(cfg.get("model_id", "")),
            port=port,
            gpu_id=gpu_id,
            extra_args=extra_args,
            enable_custom_logit_processor=enable_custom_logit_processor,
        )
