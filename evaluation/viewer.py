"""Real-time viewer for an RNN-controlled Breakout episode."""

import pygame

from environment.breakout import BreakoutEnv, FPS
from model.rnn import BreakoutRNN


def watch_model(model, seed=42):
    """Render one model playing. R restarts; N advances to the next seed."""
    pygame.init()
    env = BreakoutEnv(seed=seed, render_mode="human")
    observation = env.reset(seed=seed)
    model.reset_state()
    env.launch()

    running = True
    current_seed = int(seed)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    observation = env.reset(seed=current_seed)
                    model.reset_state()
                    env.launch()
                elif event.key == pygame.K_n:
                    current_seed += 1
                    observation = env.reset(seed=current_seed)
                    model.reset_state()
                    env.launch()

        if not env.done:
            action = model.act(observation)
            observation, _, _, _ = env.step(action)

        env.render()
        env.clock.tick(FPS)

    env.close()
    pygame.quit()
