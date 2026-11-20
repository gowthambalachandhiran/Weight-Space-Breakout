"""Gemini client that proposes bounded scalar changes to RNN weights."""

import json
import urllib.parse

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


class GeminiWeightProgrammer(BaseWeightProgrammer):
    provider_name = "Gemini"
    api_key_environment = "GEMINI_API_KEY"
    model_environment = "GEMINI_MODEL"
    default_model = config.GEMINI_MODEL

    def propose_modifications(self, weights, metrics, generation):
        model = urllib.parse.quote(self.model_name, safe="")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?"
            + urllib.parse.urlencode({"key": self.api_key})
        )
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTIONS}]},
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": json.dumps(
                        self._proposal_payload(weights, metrics, generation)
                    )
                }],
            }],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
            },
        }
        response_data = self._post_json(
            url, body, {"Content-Type": "application/json"}
        )
        try:
            raw = response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Gemini response: {response_data}"
            ) from exc
        return self._parse_proposal(raw)
