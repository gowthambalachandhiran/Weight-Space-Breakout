"""Sequential single-model generations driven by LLM weight edits."""

import json
from pathlib import Path

import config
from evaluation.evaluator import evaluate_model
from model.rnn import BreakoutRNN
from model.weights import apply_patches, load_weights, save_weights
from programmer import create_programmer


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(generations=5, llm="gemini", llm_model=None, output_dir=None):
    generations_dir = Path(output_dir) if output_dir is not None else config.GENERATIONS_DIR
    generations_dir.mkdir(parents=True, exist_ok=True)
    model = BreakoutRNN(seed=config.INITIAL_WEIGHT_SEED)
    programmer = create_programmer(llm, model_name=llm_model)
    best_score = float("-inf")
    best_path = generations_dir / "best.json"
    history = []

    print("\\nWEIGHT-SPACE BREAKOUT")
    print(f"Training seeds: {config.TRAIN_SEEDS}")
    print(f"Held-out seeds: {config.HELD_OUT_SEEDS} (not sent to the LLM)")
    print(f"LLM provider: {programmer.provider_name}")
    print(f"LLM model: {programmer.model_name}")

    # Generation 0 is the random baseline. Each following generation is one
    # Gemini proposal applied to the best checkpoint seen so far.
    for generation in range(generations + 1):
        print(f"\\n--- Generation {generation} ---")
        metrics = evaluate_model(model, config.TRAIN_SEEDS)
        generation_dir = generations_dir / f"generation_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        save_weights(model, generation_dir / "weights.json")
        write_json(generation_dir / "metrics.json", metrics)

        score = metrics[config.FITNESS_METRIC]
        print(
            f"Average score: {metrics['average_score']} | "
            f"Average steps: {metrics['average_steps']} | "
            f"Win rate: {metrics['win_rate']}"
        )
        is_best = score > best_score
        if is_best:
            best_score = score
            save_weights(model, best_path)
            print("New best checkpoint saved.")

        history.append({
            "generation": generation,
            "average_score": metrics["average_score"],
            "best_so_far": best_score,
            "is_new_best": is_best,
        })
        write_json(generations_dir / "history.json", history)

        if generation == generations:
            break

        # Revert to best-known weights, then measure that exact parent checkpoint.
        # This keeps the metrics Gemini sees aligned with the weights it is editing.
        load_weights(model, best_path)
        parent_metrics = evaluate_model(model, config.TRAIN_SEEDS)
        proposal = programmer.propose_modifications(
            weights=model.get_weights(), metrics=parent_metrics, generation=generation
        )
        print(f"{programmer.provider_name}:", proposal["reasoning"])
        applied = apply_patches(model, proposal["modifications"])
        write_json(generation_dir / f"{llm}_proposal.json", proposal)
        write_json(generation_dir / "applied_patches.json", applied)
        print(f"Applied {len(applied)} direct weight changes.")

    # Held-out seeds are used only after evolution, for final generalization check.
    load_weights(model, best_path)
    held_out = evaluate_model(model, config.HELD_OUT_SEEDS)
    write_json(generations_dir / "held_out_metrics.json", held_out)
    print("\\n--- Best checkpoint: held-out evaluation ---")
    print(f"Average score: {held_out['average_score']}")
    print(f"Win rate: {held_out['win_rate']}")
    print(f"Saved results: {generations_dir}")
    return held_out
