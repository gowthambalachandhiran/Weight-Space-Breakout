"""Inference-only vanilla RNN. No optimizer, loss, or gradient updates."""

import torch
import config


class BreakoutRNN:
    MATRIX_NAMES = ("W_xh", "W_hh", "W_hy", "b_h", "b_y")

    def __init__(self, seed=config.INITIAL_WEIGHT_SEED):
        self.input_size = config.INPUT_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.output_size = config.OUTPUT_SIZE
        self.hidden = torch.zeros(self.hidden_size, dtype=torch.float32)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.W_xh = torch.randn(self.input_size, self.hidden_size, generator=generator) * 0.35
        self.W_hh = torch.randn(self.hidden_size, self.hidden_size, generator=generator) * 0.20
        self.W_hy = torch.randn(self.hidden_size, self.output_size, generator=generator) * 0.35
        self.b_h = torch.zeros(self.hidden_size, dtype=torch.float32)
        self.b_y = torch.zeros(self.output_size, dtype=torch.float32)

    def reset_state(self):
        self.hidden.zero_()

    @torch.no_grad()
    def forward(self, observation):
        x = torch.as_tensor(observation, dtype=torch.float32)
        if x.ndim != 1 or x.numel() != self.input_size:
            raise ValueError(f"Expected {self.input_size} observation values")
        self.hidden = torch.tanh(x @ self.W_xh + self.hidden @ self.W_hh + self.b_h)
        return self.hidden @ self.W_hy + self.b_y

    @torch.no_grad()
    def act(self, observation):
        return int(torch.argmax(self.forward(observation)).item())

    def get_weights(self):
        return {name: getattr(self, name).tolist() for name in self.MATRIX_NAMES}

    def set_weights(self, weights):
        shapes = {
            "W_xh": (self.input_size, self.hidden_size),
            "W_hh": (self.hidden_size, self.hidden_size),
            "W_hy": (self.hidden_size, self.output_size),
            "b_h": (self.hidden_size,),
            "b_y": (self.output_size,),
        }
        for name, shape in shapes.items():
            if name not in weights:
                raise ValueError(f"Missing weight: {name}")
            tensor = torch.tensor(weights[name], dtype=torch.float32)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
            setattr(self, name, tensor)
        self.reset_state()
