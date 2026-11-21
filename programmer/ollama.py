"""Local Ollama client for bounded scalar changes to RNN weights."""

import json
import math
import os
from copy import deepcopy

import config
from programmer.base import BaseWeightProgrammer, SYSTEM_INSTRUCTIONS


OLLAMA_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "modifications": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "string",
                        "enum": list(config.RNN_WEIGHT_SHAPES),
                    },
                    "row": {"type": "integer"},
                    "column": {"type": "integer"},
                    "index": {"type": "integer"},
                    "delta": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
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

RETRY_MAX_PATCHES = 12


class InvalidOllamaProposalError(RuntimeError):
    """Raised when Ollama returns malformed or unusable proposal content."""


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
        try:
            self.temperature = float(os.getenv(
                "OLLAMA_TEMPERATURE", str(config.OLLAMA_TEMPERATURE)
            ))
        except ValueError as exc:
            raise ValueError("OLLAMA_TEMPERATURE must be a number.") from exc
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("OLLAMA_TEMPERATURE must be between 0 and 2.")

    def _request_proposal(self, weights, metrics, generation, retry=False):
        ollama_instructions = (
            f"{SYSTEM_INSTRUCTIONS}\n"
            "Return a focused proposal that tests a plausible behavior "
            "hypothesis; use multiple scalar edits when they are justified "
            "and do not artificially minimize their count or size within the "
            "stated limits. You cannot apply edits or "
            "claim that you have applied them. Do not return summaries, metrics, "
            "full matrices, or next steps. For each modification, include all "
            "coordinates required for that matrix: row and column for matrices, "
            "or index for biases."
        )
        proposal_schema = OLLAMA_PROPOSAL_SCHEMA
        temperature = self.temperature
        if retry:
            ollama_instructions += (
                "\nThe previous response was invalid or incomplete. This is "
                "one retry: return a complete JSON object only, with a "
                "one-sentence rationale and at most 12 modifications. Include "
                "row and column for every matrix edit or index for every bias edit."
            )
            proposal_schema = deepcopy(OLLAMA_PROPOSAL_SCHEMA)
            proposal_schema["properties"]["modifications"]["maxItems"] = min(
                RETRY_MAX_PATCHES, config.MAX_WEIGHT_PATCHES
            )
            temperature = min(temperature, 0.2)
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
            "format": proposal_schema,
            "think": False,
            "stream": False,
            "options": {"temperature": temperature},
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
            raise InvalidOllamaProposalError(
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
            raise InvalidOllamaProposalError(
                f"Ollama returned an empty final answer{suffix}. "
                "Thinking is disabled for this request; check the model/server "
                "and try again."
            )
        try:
            return self._parse_proposal(raw)
        except RuntimeError as exc:
            raise InvalidOllamaProposalError(
                f"Ollama returned an unusable proposal: {exc}"
            ) from exc

    def _validate_modifications(self, modifications):
        dimensions = config.RNN_WEIGHT_SHAPES
        accepted = []
        rejected_count = 0
        for patch in modifications:
            if (
                len(accepted) >= config.MAX_WEIGHT_PATCHES
                or not isinstance(patch, dict)
            ):
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
        return accepted, rejected_count

    def propose_modifications(self, weights, metrics, generation):
        first_failure = None
        for attempt in range(2):
            try:
                proposal = self._request_proposal(
                    weights, metrics, generation, retry=bool(attempt)
                )
            except InvalidOllamaProposalError as exc:
                if attempt == 0:
                    first_failure = exc
                    continue
                raise RuntimeError(
                    "Ollama returned an unusable proposal after one retry. "
                    f"First attempt: {first_failure}. Retry: {exc}. "
                    "The experiment stopped without applying any edits."
                ) from exc

            modifications = proposal["modifications"]
            accepted, rejected_count = self._validate_modifications(
                modifications
            )
            if modifications and not accepted:
                failure = InvalidOllamaProposalError(
                    f"all {rejected_count} proposed edits failed client validation"
                )
                if attempt == 0:
                    first_failure = failure
                    continue
                raise RuntimeError(
                    "Ollama returned no valid edits after one retry; "
                    f"the retry's {rejected_count} edits failed validation. "
                    "The experiment stopped without applying any edits."
                )

            proposal["modifications"] = accepted
            if rejected_count:
                proposal["reasoning"] += (
                    f" The client discarded {rejected_count} invalid or excessive "
                    "modification(s); only validated bounded scalar edits are eligible."
                )
            if attempt:
                proposal["reasoning"] += (
                    " The first response was invalid; this proposal was generated "
                    "on one retry."
                )
            return proposal
