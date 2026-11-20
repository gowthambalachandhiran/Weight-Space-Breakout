"""Anthropic Claude client that proposes bounded scalar changes to RNN weights."""

import json

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


class ClaudeWeightProgrammer(BaseWeightProgrammer):
    provider_name = "Claude"
    api_key_environment = "ANTHROPIC_API_KEY"
    model_environment = "ANTHROPIC_MODEL"
    default_model = config.ANTHROPIC_MODEL

    def propose_modifications(self, weights, metrics, generation):
        body = {
            "model": self.model_name,
            "max_tokens": 1200,
            "system": SYSTEM_INSTRUCTIONS,
            "messages": [{
                "role": "user",
                "content": json.dumps(
                    self._proposal_payload(weights, metrics, generation)
                ),
            }],
            "temperature": 0.4,
        }
        response_data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            body,
            {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            raw = "".join(
                block["text"]
                for block in response_data["content"]
                if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Claude response: {response_data}"
            ) from exc
        if not raw:
            raise RuntimeError(f"Claude returned no text content: {response_data}")
        return self._parse_proposal(raw)
