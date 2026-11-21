"""Shared prompt, request, and response handling for LLM providers."""

import json
import urllib.error
import urllib.request

import config


SYSTEM_INSTRUCTIONS = """
You are programming a two-layer vanilla RNN controlling Breakout. Each
recurrent layer has 16 hidden units.
Matrices: W_xh [7,16], W_hh1 [16,16], W_h1h2 [16,16],
W_hh2 [16,16], W_hy [16,3], b_h1 [16], b_h2 [16], b_y [3].
Inputs: ball_x [0,1], ball_y [0,1], ball_vx [-1,1], ball_vy [-1,1],
paddle_center_x [0,1], ball_offset_from_paddle [-1,1],
vertical_distance_to_paddle [0,1].
Actions selected by argmax: 0=left, 1=stay, 2=right.

Use the supplied fixed-training-seed metrics and weights to form a concrete,
testable hypothesis about the RNN's behavior. Propose purposeful direct scalar
changes that meaningfully test that hypothesis. Do not restrict proposals to
tiny changes or only one or a few weights: use multiple coordinated edits and
meaningful deltas when justified, within the limits below. Avoid arbitrary
changes. This is not gradient descent; do not claim a change is guaranteed to
improve performance.

Return JSON only:
{
  "reasoning": "brief explanation",
  "modifications": [
    {"matrix":"W_h1h2","row":0,"column":1,"delta":0.03},
    {"matrix":"b_y","index":2,"delta":-0.02}
  ]
}
At most 50 modifications. Each delta must be between -1.0 and +1.0.
For matrices use row and column. For bias vectors use index.
Only use W_xh, W_hh1, W_h1h2, W_hh2, W_hy, b_h1, b_h2, b_y.
No full matrices, markdown fences, code, or extra keys.
An empty modifications list is allowed if evidence is insufficient.
"""


class BaseWeightProgrammer:
    provider_name = "LLM"
    api_key_environment = ""
    model_environment = ""
    default_model = ""

    def __init__(self, model_name=None, api_key=None):
        import os

        self.api_key = api_key or os.getenv(self.api_key_environment)
        self.model_name = model_name or os.getenv(
            self.model_environment, self.default_model
        )
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_environment} is not set. Configure it in .env."
            )

    def _proposal_payload(self, weights, metrics, generation):
        return {
            "generation": int(generation),
            "architecture": {
                "input_size": config.INPUT_SIZE,
                "hidden_size": config.HIDDEN_SIZE,
                "hidden_layers": config.HIDDEN_LAYERS,
                "output_size": config.OUTPUT_SIZE,
            },
            "action_mapping": {"0": "left", "1": "stay", "2": "right"},
            "training_seed_metrics": metrics,
            "current_weights": weights,
            "instruction": (
                "Use a testable hypothesis and purposeful weight changes to "
                "improve average score. Use meaningful coordinated edits when "
                "justified, within the numeric limits, without overfitting seeds."
            ),
        }

    def _post_json(self, url, body, headers, timeout=120):
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.provider_name} API HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {self.provider_name} API: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"{self.provider_name} request timed out after {timeout} seconds."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unexpected {self.provider_name} API response: invalid JSON") from exc

    def _parse_proposal(self, raw):
        try:
            result = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            preview = repr(raw)
            if len(preview) > 500:
                preview = preview[:497] + "..."
            raise RuntimeError(
                f"{self.provider_name} returned invalid proposal JSON: {preview}"
            ) from exc
        if not isinstance(result, dict) or not isinstance(
            result.get("modifications"), list
        ):
            keys = sorted(result) if isinstance(result, dict) else type(result).__name__
            raise RuntimeError(
                f"{self.provider_name} must return a JSON object with "
                f"a modifications list; received {keys}"
            )
        return {
            "reasoning": str(result.get("reasoning", "")),
            "modifications": result["modifications"],
        }
