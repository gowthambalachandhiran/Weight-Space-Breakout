"""Local Ollama client for bounded scalar changes to RNN weights."""

import json
import os

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


class OllamaWeightProgrammer(BaseWeightProgrammer):
    provider_name = "Ollama"
    model_environment = "OLLAMA_MODEL"
    default_model = config.OLLAMA_MODEL

    def __init__(self, model_name=None, base_url=None):
        self.api_key = None
        self.model_name = model_name or os.getenv(
            self.model_environment, self.default_model
        )
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
        ).rstrip("/")
        try:
            self.timeout = int(os.getenv(
                "OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS)
            ))
        except ValueError as exc:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be an integer.") from exc
        if self.timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")

    def propose_modifications(self, weights, metrics, generation):
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": json.dumps(
                        self._proposal_payload(weights, metrics, generation)
                    ),
                },
            ],
            "format": "json",
            "think": False,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        response_data = self._post_json(
            f"{self.base_url}/api/chat",
            body,
            {"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        try:
            raw = response_data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response: {response_data}"
            ) from exc
        if not isinstance(raw, str) or not raw.strip():
            message = response_data.get("message", {})
            details = []
            if message.get("thinking"):
                details.append("the model returned thinking but no final answer")
            done_reason = response_data.get("done_reason")
            if done_reason:
                details.append(f"generation ended with reason {done_reason!r}")
            suffix = f" ({'; '.join(details)})" if details else ""
            raise RuntimeError(
                f"Ollama returned an empty final answer{suffix}. "
                "Thinking is disabled for this request; check the model/server "
                "and try again."
            )
        return self._parse_proposal(raw)
