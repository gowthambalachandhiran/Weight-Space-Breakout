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
            "stream": False,
            "options": {"temperature": 0.4},
        }
        response_data = self._post_json(
            f"{self.base_url}/api/chat",
            body,
            {"Content-Type": "application/json"},
        )
        try:
            raw = response_data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response: {response_data}"
            ) from exc
        return self._parse_proposal(raw)
