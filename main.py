"""CLI for playing Breakout, baseline evaluation, and LLM weight programming."""

import argparse
from dotenv import load_dotenv
from pathlib import Path

import config
from environment.breakout import run_local_game
from evaluation.evaluator import evaluate_model
from evaluation.viewer import watch_model
from experiments.experiment import run_experiment
from model.rnn import BreakoutRNN
from model.weights import load_weights, save_weights


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Weight Space Breakout")
    parser.add_argument("--mode", choices=("play", "watch", "baseline", "evolve", "test"),
                        default="play")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--llm", choices=("gemini", "gpt", "claude", "ollama"), default="gemini",
                        help="LLM provider used for weight proposals (default: gemini)")
    parser.add_argument("--llm-model", default=None,
                        help="Override the provider's default model")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for this run's checkpoints and metrics")
    parser.add_argument("--generation", type=int, default=None,
                        help="Generation checkpoint to watch; omit to watch best.json")
    args = parser.parse_args()

    if args.mode == "play":
        run_local_game(args.seed)
    elif args.mode == "watch":
        model = BreakoutRNN()
        generations_dir = (
            args.output_dir if args.output_dir is not None else config.GENERATIONS_DIR
        )
        if args.generation is None:
            checkpoint = generations_dir / "best.json"
        else:
            checkpoint = (
                generations_dir / f"generation_{args.generation:03d}"
                / "weights.json"
            )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Run baseline/evolution first."
            )
        load_weights(model, checkpoint)
        watch_model(model, seed=args.seed)
    elif args.mode == "baseline":
        model = BreakoutRNN()
        metrics = evaluate_model(model, config.TRAIN_SEEDS)
        baseline_dir = config.GENERATIONS_DIR / "generation_000"
        save_weights(model, baseline_dir / "weights.json")
        save_weights(model, config.GENERATIONS_DIR / "best.json")
        print(metrics)
        print(f"Baseline checkpoint saved to: {baseline_dir / 'weights.json'}")
    elif args.mode == "evolve":
        run_experiment(
            generations=args.generations,
            llm=args.llm,
            llm_model=args.llm_model,
            output_dir=args.output_dir,
        )
    elif args.mode == "test":
        checkpoint = config.GENERATIONS_DIR / "best.json"
        if not checkpoint.exists():
            raise FileNotFoundError("Run --mode evolve first; best.json is missing.")
        model = BreakoutRNN()
        load_weights(model, checkpoint)
        print(evaluate_model(model, config.HELD_OUT_SEEDS))


if __name__ == "__main__":
    main()
