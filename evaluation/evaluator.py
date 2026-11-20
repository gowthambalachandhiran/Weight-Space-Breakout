"""Headless evaluation on fixed, deterministic seeds."""

import statistics
from environment.breakout import BreakoutEnv
from config import MAX_STEPS_PER_EPISODE


def evaluate_model(model, seeds, max_steps=MAX_STEPS_PER_EPISODE):
    episodes = []
    for seed in seeds:
        env = BreakoutEnv(seed=int(seed), render_mode=None)
        model.reset_state()
        observation = env.reset(seed=int(seed))
        env.launch()
        total_reward = 0.0
        steps = 0

        while not env.done and steps < max_steps:
            action = model.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        episodes.append({
            "seed": int(seed),
            "score": int(env.score),
            "bricks_destroyed": int(env.score),
            "bricks_remaining": int(info["bricks_remaining"]),
            "steps": steps,
            "lives_remaining": int(env.lives),
            "won": bool(env.won),
            "completed": bool(env.done),
            "total_reward": round(total_reward, 4),
        })
        env.close()

    scores = [e["score"] for e in episodes]
    steps_list = [e["steps"] for e in episodes]
    rewards = [e["total_reward"] for e in episodes]
    return {
        "episodes": episodes,
        "average_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "best_score": max(scores) if scores else 0,
        "average_steps": round(statistics.mean(steps_list), 2) if steps_list else 0.0,
        "average_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "win_rate": round(sum(e["won"] for e in episodes) / len(episodes), 4) if episodes else 0.0,
        "seeds": list(seeds),
    }
