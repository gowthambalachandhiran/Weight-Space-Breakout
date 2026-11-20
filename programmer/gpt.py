"""OpenAI GPT client that proposes bounded scalar changes to RNN weights."""

import json

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


class GPTWeightProgrammer(BaseWeightProgrammer):
    provider_name = "OpenAI"
    api_key_environment = "OPENAI_API_KEY"
    model_environment = "OPENAI_MODEL"
    default_model = config.OPENAI_MODEL

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
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        response_data = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            raw = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI response: {response_data}"
            ) from exc
        return self._parse_proposal(raw)
