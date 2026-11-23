"""JSON checkpoints and bounded direct weight modifications."""

import json
from pathlib import Path
import torch
import config


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "hidden_layers": model.hidden_layers,
            "output_size": model.output_size,
        },
        "weights": model.get_weights(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_weights(model, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "output_size": model.output_size,
    }
    if model.hidden_layers != 1:
        expected["hidden_layers"] = model.hidden_layers
    if payload.get("architecture") != expected:
        raise ValueError("Checkpoint architecture does not match this RNN")
    model.set_weights(payload["weights"])
    return model


def apply_patches(model, modifications):
    if not isinstance(modifications, list):
        raise ValueError("modifications must be a list")
    if len(modifications) > config.MAX_WEIGHT_PATCHES:
        raise ValueError("LLM proposed too many patches")

    applied = []
    with torch.no_grad():
        for patch in modifications:
            name = patch.get("matrix")
            if name not in model.MATRIX_NAMES:
                raise ValueError(f"Unknown matrix: {name}")
            delta = float(patch["delta"])
            if not torch.isfinite(torch.tensor(delta)) or abs(delta) > config.MAX_PATCH_DELTA:
                raise ValueError(f"Invalid or excessive delta: {delta}")

            tensor = getattr(model, name)
            if tensor.ndim == 1:
                index = int(patch["index"])
                if not 0 <= index < tensor.shape[0]:
                    raise ValueError(f"Index out of range for {name}")
                old = float(tensor[index].item())
                new = max(-config.WEIGHT_ABS_LIMIT, min(config.WEIGHT_ABS_LIMIT, old + delta))
                tensor[index] = new
                applied.append({"matrix": name, "index": index, "delta": delta, "old": old, "new": new})
            else:
                row, col = int(patch["row"]), int(patch["column"])
                if not (0 <= row < tensor.shape[0] and 0 <= col < tensor.shape[1]):
                    raise ValueError(f"Index out of range for {name}")
                old = float(tensor[row, col].item())
                new = max(-config.WEIGHT_ABS_LIMIT, min(config.WEIGHT_ABS_LIMIT, old + delta))
                tensor[row, col] = new
                applied.append({"matrix": name, "row": row, "column": col, "delta": delta, "old": old, "new": new})

    model.reset_state()
    return applied
