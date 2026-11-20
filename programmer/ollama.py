"""Local Ollama client for bounded scalar changes to RNN weights."""

import json
import math
import os

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


OLLAMA_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "modifications": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "string",
                        "enum": ["W_xh", "W_hh", "W_hy", "b_h", "b_y"],
                    },
                    "row": {"type": "integer"},
                    "column": {"type": "integer"},
                    "index": {"type": "integer"},
                    "delta": {
                        "type": "number",
                        "minimum": -0.15,
                        "maximum": 0.15,
                    },
                },
                "required": ["matrix", "delta"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "modifications"],
    "additionalProperties": False,
}


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
        ollama_instructions = (
            f"{SYSTEM_INSTRUCTIONS}\n"
            "Return only a conservative proposal; you cannot apply edits or "
            "claim that you have applied them. Do not return summaries, metrics, "
            "full matrices, or next steps. For each modification, include all "
            "coordinates required for that matrix: row and column for matrices, "
            "or index for biases. Prefer a small number of individual scalar edits."
        )
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": ollama_instructions},
                {
                    "role": "user",
                    "content": json.dumps(
                        self._proposal_payload(weights, metrics, generation)
                    ),
                },
            ],
            "format": OLLAMA_PROPOSAL_SCHEMA,
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
        try:
            proposal = self._parse_proposal(raw)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Ollama returned an unusable proposal: {exc}. "
                "The experiment stopped without applying any edits."
            ) from exc
        dimensions = {
            "W_xh": (7, 16),
            "W_hh": (16, 16),
            "W_hy": (16, 3),
            "b_h": (16,),
            "b_y": (3,),
        }
        accepted = []
        rejected_count = 0
        for patch in proposal["modifications"]:
            if len(accepted) >= config.MAX_WEIGHT_PATCHES or not isinstance(patch, dict):
                rejected_count += 1
                continue
            matrix = patch.get("matrix")
            delta = patch.get("delta")
            if (
                not isinstance(matrix, str)
                or matrix not in dimensions
                or isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or (isinstance(delta, float) and not math.isfinite(delta))
                or abs(delta) > config.MAX_PATCH_DELTA
            ):
                rejected_count += 1
                continue
            if len(dimensions[matrix]) == 1:
                index = patch.get("index")
                if type(index) is not int or not 0 <= index < dimensions[matrix][0]:
                    rejected_count += 1
                    continue
                accepted.append({
                    "matrix": matrix,
                    "index": index,
                    "delta": delta,
                })
            else:
                row, column = patch.get("row"), patch.get("column")
                if (
                    type(row) is not int
                    or type(column) is not int
                    or not 0 <= row < dimensions[matrix][0]
                    or not 0 <= column < dimensions[matrix][1]
                ):
                    rejected_count += 1
                    continue
                accepted.append({
                    "matrix": matrix,
                    "row": row,
                    "column": column,
                    "delta": delta,
                })
        proposal["modifications"] = accepted
        if rejected_count:
            proposal["reasoning"] += (
                f" The client discarded {rejected_count} invalid or excessive "
                "modification(s); only validated bounded scalar edits are eligible."
            )
        return proposal
