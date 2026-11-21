import unittest
import sys
from unittest.mock import patch
from environment.breakout import BreakoutEnv
from model.rnn import BreakoutRNN
from model.weights import apply_patches
from programmer import create_programmer


class ProjectTests(unittest.TestCase):
    def test_watch_uses_custom_output_dir_for_generation(self):
        import main

        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--mode",
                "watch",
                "--output-dir",
                r"generations\ollama-qwen3.5-4b",
                "--generation",
                "5",
            ],
        ), patch.object(main, "BreakoutRNN"), patch.object(
            main, "load_weights"
        ) as load_weights, patch.object(main, "watch_model"):
            main.main()

        self.assertEqual(
            load_weights.call_args[0][1],
            main.Path(r"generations\ollama-qwen3.5-4b")
            / "generation_005"
            / "weights.json",
        )

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

    def test_direct_patch_accepts_delta_limit(self):
        model = BreakoutRNN(seed=9)
        applied = apply_patches(model, [{
            "matrix": "b_y", "index": 0, "delta": 1.0
        }])
        self.assertEqual(applied[0]["delta"], 1.0)

    def test_direct_patch_rejects_delta_above_limit(self):
        model = BreakoutRNN(seed=9)
        with self.assertRaisesRegex(ValueError, "Invalid or excessive delta"):
            apply_patches(model, [{
                "matrix": "b_y", "index": 0, "delta": 1.01
            }])

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
        self.assertEqual(programmer.temperature, 0.7)

    def test_ollama_temperature_can_be_configured(self):
        with patch.dict("os.environ", {"OLLAMA_TEMPERATURE": "1.1"}):
            programmer = create_programmer("ollama")
        self.assertEqual(programmer.temperature, 1.1)

    def test_ollama_rejects_invalid_temperature(self):
        for value in ("not-a-number", "nan", "-0.1", "2.1"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"OLLAMA_TEMPERATURE": value}
            ):
                with self.assertRaisesRegex(
                    ValueError, "OLLAMA_TEMPERATURE"
                ):
                    create_programmer("ollama")

    def test_ollama_timeout_can_be_configured(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "900"}):
            programmer = create_programmer("ollama")
        self.assertEqual(programmer.timeout, 900)

    def test_ollama_parses_json_chat_response(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": (
                    '{"reasoning":"small adjustment","modifications":['
                    '{"matrix":"W_hy","row":0,"column":1,"delta":0.03}]}'
                )
            }
        }
        with patch.object(programmer, "_post_json", return_value=response):
            proposal = programmer.propose_modifications({}, {}, 0)
        self.assertEqual(proposal, {
            "reasoning": "small adjustment",
            "modifications": [{
                "matrix": "W_hy",
                "row": 0,
                "column": 1,
                "delta": 0.03,
            }],
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
        self.assertEqual(body["options"]["temperature"], programmer.temperature)
        self.assertEqual(body["format"]["type"], "object")
        self.assertIn("modifications", body["format"]["required"])
        self.assertEqual(body["format"]["properties"]["modifications"]["maxItems"], 50)
        delta_schema = body["format"]["properties"]["modifications"]["items"][
            "properties"
        ]["delta"]
        self.assertEqual(delta_schema["minimum"], -1.0)
        self.assertEqual(delta_schema["maximum"], 1.0)

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

    def test_ollama_rejects_non_proposal_json_without_applying(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": '{"status":"success","modified_weights_preview":{}}'
            }
        }
        with patch.object(programmer, "_post_json", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "unusable proposal.*modifications list.*without applying",
            ):
                programmer.propose_modifications({}, {}, 0)

    def test_ollama_discards_invalid_patch_fields(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": (
                    '{"reasoning":"edit","modifications":['
                    '{"matrix":"W_hy","row":null,"column":null,"delta":0.03}]}'
                )
            }
        }
        with patch.object(programmer, "_post_json", return_value=response):
            proposal = programmer.propose_modifications({}, {}, 0)
        self.assertEqual(proposal["modifications"], [])
        self.assertIn("discarded 1 invalid", proposal["reasoning"])

    def test_ollama_discards_out_of_range_deltas(self):
        programmer = create_programmer("ollama", model_name="test-local-model")
        response = {
            "message": {
                "content": (
                    '{"reasoning":"edit","modifications":['
                    '{"matrix":"W_hy","row":0,"column":1,"delta":1.1}]}'
                )
            }
        }
        with patch.object(programmer, "_post_json", return_value=response):
            proposal = programmer.propose_modifications({}, {}, 0)
        self.assertEqual(proposal["modifications"], [])
        self.assertIn("discarded 1 invalid", proposal["reasoning"])

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
