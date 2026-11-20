import unittest
from unittest.mock import patch
from environment.breakout import BreakoutEnv
from model.rnn import BreakoutRNN
from model.weights import apply_patches
from programmer import create_programmer


class ProjectTests(unittest.TestCase):
    def test_observation_has_seven_values(self):
        env = BreakoutEnv(seed=7)
        self.assertEqual(len(env.get_observation()), 7)

    def test_seeded_replay_is_deterministic(self):
        actions = [1, 0, 0, 2, 2, 1, 0, 2] * 10

        def replay():
            env = BreakoutEnv(seed=123)
            env.reset(seed=123)
            env.launch()
            result = []
            for action in actions:
                obs, _, done, _ = env.step(action)
                result.append(obs)
                if done:
                    break
            return result

        self.assertEqual(replay(), replay())

    def test_rnn_returns_valid_action(self):
        model = BreakoutRNN(seed=9)
        obs = (0.5, 0.7, 0.1, -0.8, 0.5, 0.0, 0.1)
        self.assertIn(model.act(obs), (0, 1, 2))

    def test_direct_patch_changes_weight(self):
        model = BreakoutRNN(seed=9)
        before = float(model.W_hy[0, 0].item())
        apply_patches(model, [{
            "matrix": "W_hy", "row": 0, "column": 0, "delta": 0.05
        }])
        after = float(model.W_hy[0, 0].item())
        self.assertAlmostEqual(after - before, 0.05, places=5)

    def test_selects_configured_llm_provider_and_model(self):
        keys = {
            "GEMINI_API_KEY": "test-gemini-key",
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        }
        with patch.dict("os.environ", keys):
            providers = {
                "gemini": ("Gemini", "GEMINI_API_KEY"),
                "gpt": ("OpenAI", "OPENAI_API_KEY"),
                "claude": ("Claude", "ANTHROPIC_API_KEY"),
            }
            for name, (provider_name, key_name) in providers.items():
                with self.subTest(provider=name):
                    programmer = create_programmer(name, model_name="test-model")
                    self.assertEqual(programmer.provider_name, provider_name)
                    self.assertEqual(programmer.model_name, "test-model")
                    self.assertEqual(programmer.api_key, keys[key_name])

    def test_selects_local_ollama_without_api_key(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        self.assertEqual(programmer.provider_name, "Ollama")
        self.assertEqual(programmer.model_name, "test-local-model")
        self.assertIsNone(programmer.api_key)
        self.assertEqual(programmer.timeout, 600)

    def test_ollama_timeout_can_be_configured(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "900"}):
            programmer = create_programmer("ollama")
        self.assertEqual(programmer.timeout, 900)

    def test_ollama_parses_json_chat_response(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": '{"reasoning":"small adjustment","modifications":[]}'
            }
        }
        with patch.object(programmer, "_post_json", return_value=response):
            proposal = programmer.propose_modifications({}, {}, 0)
        self.assertEqual(proposal, {
            "reasoning": "small adjustment",
            "modifications": [],
        })

    def test_ollama_disables_thinking_for_json_proposals(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": '{"reasoning":"small adjustment","modifications":[]}'
            }
        }
        with patch.object(
            programmer, "_post_json", return_value=response
        ) as post_json:
            programmer.propose_modifications({}, {}, 0)
        body = post_json.call_args.args[1]
        self.assertIs(body["think"], False)
        self.assertEqual(body["format"], "json")

    def test_ollama_empty_final_answer_has_clear_error(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": "",
                "thinking": "internal trace",
            },
            "done_reason": "length",
        }
        with patch.object(programmer, "_post_json", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "empty final answer .*thinking but no final answer",
            ):
                programmer.propose_modifications({}, {}, 0)

    def test_ollama_request_uses_configured_timeout(self):
        programmer = create_programmer("ollama")
        response = {
            "message": {
                "content": '{"reasoning":"small adjustment","modifications":[]}'
            }
        }
        with patch.object(
            programmer, "_post_json", return_value=response
        ) as post_json:
            programmer.propose_modifications({}, {}, 0)
        self.assertEqual(post_json.call_args.kwargs["timeout"], 600)

    def test_request_timeout_has_clear_error(self):
        programmer = create_programmer("ollama")
        with patch(
            "programmer.base.urllib.request.urlopen",
            side_effect=TimeoutError,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Ollama request timed out after 17 seconds"
            ):
                programmer._post_json("http://localhost", {}, {}, timeout=17)

    def test_rejects_unknown_llm_provider(self):
        with self.assertRaisesRegex(ValueError, "Unknown LLM provider"):
            create_programmer("unknown")


if __name__ == "__main__":
    unittest.main()
