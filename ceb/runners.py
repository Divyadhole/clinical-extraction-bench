"""Model adapters.

Ollama by default: no API key, no bill, runs on the laptop. The hosted adapter
is here and wired up but switched off, so adding a paid model later is a
config line rather than a rewrite. That comparison is worth doing eventually -
'the hosted model is better but costs this much' is the finding employers care
about - but it is not needed to publish v1.
"""

import os
import time
from typing import Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class Result:
    def __init__(self, text: str, latency_s: float, error: Optional[str] = None,
                 prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.latency_s = latency_s
        self.error = error
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class OllamaRunner:
    """Local model through Ollama. Cost is zero by construction."""

    kind = "local"
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def __init__(self, model: str, timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> Result:
        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # Deterministic: a benchmark whose numbers move between runs
                    # is not a benchmark.
                    "options": {"temperature": 0, "seed": 7, "num_predict": 256},
                },
                timeout=self.timeout,
            )
            elapsed = time.perf_counter() - started
            resp.raise_for_status()
            payload = resp.json()
            return Result(
                payload.get("response", ""),
                elapsed,
                prompt_tokens=payload.get("prompt_eval_count", 0),
                completion_tokens=payload.get("eval_count", 0),
            )
        except Exception as exc:
            return Result("", time.perf_counter() - started, error=str(exc)[:200])


class EchoRunner:
    """Deterministic fake used by the tests. Never hits a network."""

    kind = "fake"
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def __init__(self, model: str = "echo", scripted=None):
        self.model = model
        self.scripted = scripted or []
        self.calls = 0

    def generate(self, prompt: str) -> Result:
        text = self.scripted[self.calls % len(self.scripted)] if self.scripted else "{}"
        self.calls += 1
        return Result(text, 0.01, prompt_tokens=10, completion_tokens=5)


def build(spec: str):
    """'ollama:qwen2.5:3b' -> OllamaRunner('qwen2.5:3b')."""
    if spec.startswith("ollama:"):
        return OllamaRunner(spec.split(":", 1)[1])
    if spec.startswith("echo"):
        return EchoRunner()
    raise ValueError(
        f"Unknown model spec {spec!r}. Use ollama:<name>, e.g. ollama:qwen2.5:3b"
    )
