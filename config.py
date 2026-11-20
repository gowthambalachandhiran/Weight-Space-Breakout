from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATIONS_DIR = PROJECT_ROOT / "generations"

INPUT_SIZE = 7
HIDDEN_SIZE = 16
OUTPUT_SIZE = 3
INITIAL_WEIGHT_SEED = 1234

TRAIN_SEEDS = [11, 22, 33]
HELD_OUT_SEEDS = [101, 202, 303]
MAX_STEPS_PER_EPISODE = 5000

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_WEIGHT_PATCHES = 12
MAX_PATCH_DELTA = 0.15
WEIGHT_ABS_LIMIT = 3.0
FITNESS_METRIC = "average_score"
