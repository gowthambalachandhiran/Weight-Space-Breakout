"""LLM-backed direct weight proposal clients."""

from programmer.claude import ClaudeWeightProgrammer
from programmer.gemini import GeminiWeightProgrammer
from programmer.gpt import GPTWeightProgrammer
from programmer.ollama import OllamaWeightProgrammer


def create_programmer(provider, model_name=None):
    programmers = {
        "gemini": GeminiWeightProgrammer,
        "gpt": GPTWeightProgrammer,
        "claude": ClaudeWeightProgrammer,
        "ollama": OllamaWeightProgrammer,
    }
    try:
        programmer_class = programmers[provider.lower()]
    except (AttributeError, KeyError) as exc:
        choices = ", ".join(programmers)
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from: {choices}") from exc
    return programmer_class(model_name=model_name)
