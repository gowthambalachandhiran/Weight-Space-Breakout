"""
Standalone Breakout environment for weight-space programming experiments.

Install:
    python -m pip install pygame

Run:
    python breakout.py
    python breakout.py --seed 42

Controls:
    Left / Right arrows : Move paddle
    Space                : Launch or restart the ball
    R                    : Restart using the current seed
    N                    : Restart with the next seed
    Esc                  : Quit

RNN action interface:
    0 = move left
    1 = stay
    2 = move right

The BreakoutEnv class can also be imported and controlled programmatically:
    env = BreakoutEnv(seed=42)
    observation = env.reset()
    observation, reward, done, info = env.step(action)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pygame


# ----------------------------- Configuration -----------------------------

WIDTH = 800
HEIGHT = 600
FPS = 60

PADDLE_WIDTH = 110
PADDLE_HEIGHT = 14
PADDLE_SPEED = 7.0
PADDLE_Y = HEIGHT - 48

BALL_RADIUS = 8
BALL_SPEED = 5.0

BRICK_ROWS = 5
BRICK_COLS = 10
BRICK_WIDTH = 68
BRICK_HEIGHT = 22
BRICK_GAP = 8
BRICK_TOP = 75
BRICK_LEFT = 28

BACKGROUND = (17, 22, 35)
WHITE = (240, 243, 250)
MUTED = (160, 171, 190)
PADDLE_COLOR = (76, 190, 255)
BALL_COLOR = (255, 224, 112)
BRICK_COLORS = [
    (255, 107, 107),
    (255, 159, 67),
    (255, 206, 84),
    (72, 199, 142),
    (84, 160, 255),
]


@dataclass
class Brick:
    rect: pygame.Rect
    alive: bool = True


class BreakoutEnv:
    """
    Small deterministic Breakout environment.

    Observations contain seven normalized values:
      0. ball_x                 [0, 1]
      1. ball_y                 [0, 1]
      2. ball_vx                [-1, 1] relative to BALL_SPEED
      3. ball_vy                [-1, 1] relative to BALL_SPEED
      4. paddle_center_x        [0, 1]
      5. horizontal_offset     [-1, 1], ball relative to paddle
      6. vertical_distance      [0, 1], ball-to-paddle vertical distance

    Actions:
      0 = left
      1 = stay
      2 = right

    The environment does not perform any learning or weight updates.
    """

    def __init__(self, seed: int = 1, render_mode: Optional[str] = None):
        self.seed = int(seed)
        self.render_mode = render_mode
        self.rng = random.Random(self.seed)
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.reset(seed=self.seed)

    def _make_bricks(self):
        self.bricks = []
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                x = BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                self.bricks.append(
                    Brick(pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT))
                )

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)

        # Reinitialize the RNG so the same seed reproduces the same episode.
        self.rng = random.Random(self.seed)

        self.paddle_x = (WIDTH - PADDLE_WIDTH) / 2
        self.ball_x = WIDTH / 2
        self.ball_y = PADDLE_Y - BALL_RADIUS - 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.launched = False

        self.score = 0
        self.lives = 3
        self.steps = 0
        self.done = False
        self.won = False
        self._make_bricks()

        return self.get_observation()

    def _serve(self):
        # Seed-controlled initial direction. Avoid nearly vertical serves.
        angle_degrees = self.rng.uniform(-55.0, 55.0)
        angle = math.radians(angle_degrees)
        self.ball_vx = BALL_SPEED * math.sin(angle)
        self.ball_vy = -BALL_SPEED * math.cos(angle)
        self.launched = True

    def get_observation(self) -> Tuple[float, ...]:
        """Return the seven normalized features consumed by the RNN."""
        paddle_center = self.paddle_x + PADDLE_WIDTH / 2
        dx = (self.ball_x - paddle_center) / (PADDLE_WIDTH / 2)
        dy = (PADDLE_Y - self.ball_y) / HEIGHT

        return (
            self.ball_x / WIDTH,
            self.ball_y / HEIGHT,
            max(-1.0, min(1.0, self.ball_vx / BALL_SPEED)),
            max(-1.0, min(1.0, self.ball_vy / BALL_SPEED)),
            paddle_center / WIDTH,
            max(-1.0, min(1.0, dx)),
            max(0.0, min(1.0, dy)),
        )

    def _move_paddle(self, action: int):
        if action == 0:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED
        elif action != 1:
            raise ValueError("action must be 0 (left), 1 (stay), or 2 (right)")

        self.paddle_x = max(0, min(WIDTH - PADDLE_WIDTH, self.paddle_x))

    def step(self, action: int):
        """
        Advance one frame.

        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self.get_observation(), 0.0, True, self._info()

        self.steps += 1
        self._move_paddle(action)

        # Keep the ball attached to the paddle until launch.
        if not self.launched:
            self.ball_x = self.paddle_x + PADDLE_WIDTH / 2
            self.ball_y = PADDLE_Y - BALL_RADIUS - 2
            return self.get_observation(), 0.0, False, self._info()

        reward = 0.0
        previous_score = self.score

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Side walls.
        if self.ball_x - BALL_RADIUS <= 0:
            self.ball_x = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + BALL_RADIUS >= WIDTH:
            self.ball_x = WIDTH - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)

        # Ceiling.
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)

        # Paddle collision: only when travelling downward.
        paddle_rect = pygame.Rect(
            int(self.paddle_x), int(PADDLE_Y), PADDLE_WIDTH, PADDLE_HEIGHT
        )
        ball_rect = pygame.Rect(
            int(self.ball_x - BALL_RADIUS),
            int(self.ball_y - BALL_RADIUS),
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

        if self.ball_vy > 0 and ball_rect.colliderect(paddle_rect):
            self.ball_y = PADDLE_Y - BALL_RADIUS
            hit_offset = (
                (self.ball_x - (self.paddle_x + PADDLE_WIDTH / 2))
                / (PADDLE_WIDTH / 2)
            )
            hit_offset = max(-1.0, min(1.0, hit_offset))

            # A small seeded jitter is the second source of randomness.
            jitter = self.rng.uniform(-0.06, 0.06)
            angle = (hit_offset * 55.0) + math.degrees(jitter)
            angle = max(-65.0, min(65.0, angle))
            angle_rad = math.radians(angle)

            self.ball_vx = BALL_SPEED * math.sin(angle_rad)
            self.ball_vy = -BALL_SPEED * math.cos(angle_rad)
            reward += 0.2

        # Brick collisions.
        for brick in self.bricks:
            if not brick.alive:
                continue
            if ball_rect.colliderect(brick.rect):
                brick.alive = False
                self.score += 1
                reward += 1.0

                # Basic collision response based on the previous ball position.
                previous_x = self.ball_x - self.ball_vx
                previous_y = self.ball_y - self.ball_vy
                previous_rect = pygame.Rect(
                    int(previous_x - BALL_RADIUS),
                    int(previous_y - BALL_RADIUS),
                    BALL_RADIUS * 2,
                    BALL_RADIUS * 2,
                )
                if previous_rect.right <= brick.rect.left or previous_rect.left >= brick.rect.right:
                    self.ball_vx *= -1
                else:
                    self.ball_vy *= -1
                break

        # Lost a life.
        if self.ball_y - BALL_RADIUS > HEIGHT:
            self.lives -= 1
            reward -= 1.0
            if self.lives <= 0:
                self.done = True
            else:
                self.launched = False
                self.ball_vx = 0.0
                self.ball_vy = 0.0
                self.ball_x = self.paddle_x + PADDLE_WIDTH / 2
                self.ball_y = PADDLE_Y - BALL_RADIUS - 2

        # Win condition.
        if all(not brick.alive for brick in self.bricks):
            self.done = True
            self.won = True
            reward += 10.0

        if self.score > previous_score:
            reward += 0.0  # Brick reward already added above.

        return self.get_observation(), reward, self.done, self._info()

    def _info(self) -> Dict:
        return {
            "seed": self.seed,
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
            "bricks_remaining": sum(brick.alive for brick in self.bricks),
            "launched": self.launched,
            "won": self.won,
        }

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Weight Space Breakout — Environment Preview")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 22, bold=True)
            self.small_font = pygame.font.SysFont("arial", 16)

        self.screen.fill(BACKGROUND)

        # Top information bar.
        title = self.font.render("WEIGHT SPACE BREAKOUT", True, WHITE)
        self.screen.blit(title, (24, 18))

        stats = self.small_font.render(
            f"Seed: {self.seed}    Score: {self.score}    Lives: {self.lives}",
            True,
            MUTED,
        )
        self.screen.blit(stats, (24, 45))

        # Bricks.
        for brick in self.bricks:
            if brick.alive:
                row = max(0, min(BRICK_ROWS - 1, (brick.rect.y - BRICK_TOP) // (BRICK_HEIGHT + BRICK_GAP)))
                pygame.draw.rect(
                    self.screen,
                    BRICK_COLORS[row],
                    brick.rect,
                    border_radius=5,
                )

        # Paddle and ball.
        paddle_rect = pygame.Rect(
            int(self.paddle_x), int(PADDLE_Y), PADDLE_WIDTH, PADDLE_HEIGHT
        )
        pygame.draw.rect(self.screen, PADDLE_COLOR, paddle_rect, border_radius=7)
        pygame.draw.circle(
            self.screen,
            BALL_COLOR,
            (int(self.ball_x), int(self.ball_y)),
            BALL_RADIUS,
        )

        # Status / controls.
        if not self.launched and not self.done:
            message = "Press SPACE to launch"
            text = self.font.render(message, True, WHITE)
            self.screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT - 105)))

        controls = self.small_font.render(
            "← → Move   SPACE Launch   R Restart   N Next seed   ESC Quit",
            True,
            MUTED,
        )
        self.screen.blit(controls, (24, HEIGHT - 25))

        if self.done:
            message = "YOU WIN!" if self.won else "GAME OVER"
            overlay = self.font.render(message, True, WHITE)
            self.screen.blit(
                overlay, overlay.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            )
            sub = self.small_font.render("Press R to play again", True, MUTED)
            self.screen.blit(
                sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 32))
            )

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None


def run_local_game(seed: int):
    pygame.init()
    env = BreakoutEnv(seed=seed, render_mode="human")
    env.render()

    running = True
    paused = False

    while running:
        action = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not env.done:
                    if not env.launched:
                        env._serve()
                elif event.key == pygame.K_r:
                    env.reset(seed=env.seed)
                elif event.key == pygame.K_n:
                    env.reset(seed=env.seed + 1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2

        # Manual mode advances only after the ball has been launched.
        if not env.done:
            env.step(action)

        env.render()
        env.clock.tick(FPS)

    env.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Run the local Breakout environment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    args = parser.parse_args()
    run_local_game(args.seed)


if __name__ == "__main__":
    main()
