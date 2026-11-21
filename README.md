# Weight Space Breakout

A small experiment testing whether GPT, Gemini, Claude, or a local Ollama
model can improve a recurrent neural network by directly modifying its weights.

**No gradient descent, backpropagation, optimizer, or RL weight updates are
used.** The RNN performs inference only. The selected LLM proposes bounded
scalar changes to its existing weight matrices.

## Architecture

- Pygame Breakout environment
- 7 normalized numerical observations
- Vanilla Elman RNN: 7 inputs → 16 hidden units → 3 outputs
- Actions: `0 = left`, `1 = stay`, `2 = right`
- The selected LLM proposes direct weight modifications
- JSON checkpoints preserve every generation

Matrices:
- `W_xh`: input-to-hidden `[7,16]`
- `W_hh`: hidden-to-hidden `[16,16]`
- `W_hy`: hidden-to-output `[16,3]`
- `b_h`: hidden bias `[16]`
- `b_y`: output bias `[3]`

## 1. Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, use the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Play Breakout locally

```powershell
python main.py --mode play --seed 42
```

Or run the environment file directly:

```powershell
python environment\breakout.py --seed 42
```

Controls: arrows move the paddle, Space launches, R restarts the same seed,
N advances the seed, Esc quits.

## 3. Watch an RNN play

After running a baseline or evolution, watch the best saved model:

```powershell
python main.py --mode watch --seed 42
```

Watch a particular generation checkpoint:

```powershell
python main.py --mode watch --generation 2 --seed 42
```

For a run saved in a custom output directory, pass that directory as well:

```powershell
python main.py --mode watch --output-dir generations\ollama-qwen3.5-4b --generation 5 --seed 42
```

In the viewer, R restarts the current seed, N advances to the next seed,
and Esc quits. The RNN controls the paddle automatically.

## 4. Run a baseline (no API key required)

```powershell
python main.py --mode baseline
```

This evaluates the randomly initialized RNN on the fixed training seeds.

## 5. Configure LLM providers

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Add the API key for each cloud provider you want to test to `.env`. Only the
selected cloud provider needs a key. Ollama runs locally and does not need an
API key.

### Try a local model with Ollama

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download).
2. Open PowerShell and download a model. The project default is:

   ```powershell
   ollama pull qwen2.5:7b-instruct
   ```

   You can use another Ollama model instead, such as `llama3.1:8b`.
3. Make sure the Ollama service is running, then start the experiment:

   ```powershell
   python main.py --mode evolve --llm ollama --generations 3 --output-dir generations\ollama
   ```

Ollama's default server address is `http://localhost:11434`. Set
`OLLAMA_BASE_URL` in `.env` if your server uses a different address. The
default local model is `qwen2.5:7b-instruct`; change `OLLAMA_MODEL` in `.env`
or pass `--llm-model` on the command line to select another model. The prompt
and model request are sent to the configured Ollama server; no cloud-provider
API key is used for this option. Local inference may take longer than a cloud
request: the default request timeout is 600 seconds (10 minutes). Increase
`OLLAMA_TIMEOUT_SECONDS` in `.env` if your machine needs more time. Ollama
proposals are requested in a constrained JSON format; edits with invalid
indices or deltas beyond the configured safety limit are discarded and
reported rather than applied. The shared prompt encourages meaningful,
hypothesis-driven proposals and coordinated edits instead of artificially
minimizing changes. Safety limits cap each proposal at 50 edits and each delta
to `[-1.0, +1.0]`; these limits apply to all LLM providers. The Ollama
sampling temperature defaults to `0.7` (previously `0.4`); set
`OLLAMA_TEMPERATURE` in `.env` to a value from `0` to `2` to adjust it. A
higher temperature can make proposals more varied, but does not make the model
reason more deeply or guarantee a better score.

Cloud model defaults and Ollama settings can also be overridden in `.env`:
`GEMINI_MODEL`, `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `OLLAMA_MODEL`, and
`OLLAMA_BASE_URL`, `OLLAMA_TEMPERATURE`, and `OLLAMA_TIMEOUT_SECONDS`. Defaults
are listed in `config.py`.

## 6. Run sequential generations with a chosen LLM

```powershell
python main.py --mode evolve --llm gemini --generations 3 --output-dir generations\gemini
python main.py --mode evolve --llm gpt --generations 3 --output-dir generations\gpt
python main.py --mode evolve --llm claude --generations 3 --output-dir generations\claude
python main.py --mode evolve --llm ollama --generations 3 --output-dir generations\ollama
```

Use `--llm-model` to override the selected provider's default model, for example
`--llm gpt --llm-model gpt-4o`. Supported `--llm` values are `gemini`, `gpt`,
`claude`, and `ollama`. To select a different local model, use:

```powershell
python main.py --mode evolve --llm ollama --llm-model llama3.1:8b --generations 3 --output-dir generations\ollama-llama
```

Without `--output-dir`, runs use the shared `generations/` directory. Use a
different output directory for each provider/model comparison so runs do not
overwrite each other's checkpoints or metrics. Each run writes its own
`best.json`, history, generation data, and held-out evaluation in that
directory.

Generation 0 is the random baseline. Each subsequent generation is one
LLM proposal applied to the best checkpoint so far. Only one model
candidate is evaluated per generation; there is no population or parallel
candidate search.

Every generation:
1. Evaluate on the same training seeds.
2. Save weights and metrics.
3. Preserve the best checkpoint.
4. Ask the selected LLM for bounded scalar changes.
5. Validate and apply the changes.
6. Continue from the best-known checkpoint.

Held-out seeds are not sent to the LLM. They are used for final evaluation.

## 7. Evaluate the saved best model

```powershell
python main.py --mode test
```

The evolution command also performs this held-out evaluation automatically.

## Output

`generations/` contains:
- `best.json`: best model weights and architecture
- `history.json`: score history by generation
- `held_out_metrics.json`: final held-out evaluation
- `generation_NNN/weights.json`: weights before that generation's LLM edit
- `generation_NNN/metrics.json`: training metrics
- `generation_NNN/{provider}_proposal.json`: the LLM's proposal
- `generation_NNN/applied_patches.json`: validated changes actually applied

## Observation vector

| Index | Feature | Range |
|---:|---|---|
| 0 | Ball X | `[0,1]` |
| 1 | Ball Y | `[0,1]` |
| 2 | Ball horizontal velocity | `[-1,1]` |
| 3 | Ball vertical velocity | `[-1,1]` |
| 4 | Paddle center X | `[0,1]` |
| 5 | Ball horizontal offset from paddle | `[-1,1]` |
| 6 | Vertical distance to paddle | `[0,1]` |

Example programmatic use:

```python
from environment.breakout import BreakoutEnv

env = BreakoutEnv(seed=42)
observation = env.reset()
env.launch()
observation, reward, done, info = env.step(1)
```

This is a research prototype, not evidence that direct weight programming
will necessarily improve performance. A larger held-out seed set and repeated
runs are needed before drawing generalization conclusions.
