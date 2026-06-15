"""
train.py — Training pipeline for the Self-Driving Car DQN Agent.
Run: python train.py --episodes 1000 --save-every 100
"""

import os
import json
import time
import argparse
import random
import math

from environment.car import Car
from environment.road import Road
from environment.sensor import Sensor
from environment.obstacle import Obstacle
from agent.dqn_agent import DQNAgent


def parse_train_args():
    p = argparse.ArgumentParser(description="Train DQN Agent")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="models")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def build_state(sensors, car):
    readings = [s.read_flat() for s in sensors]
    return readings + [
        car.speed / 120.0,
        car.lane / 2.0,
        car.position / 1000.0,
        car.heading / 180.0
    ]


def compute_reward(car, obstacles, road):
    reward = 0.1

    # Speed band reward
    if 30 <= car.speed <= 70:
        reward += 0.4
    elif car.speed < 10:
        reward -= 0.1

    # Center-lane preference
    if car.lane == 1:
        reward += 0.05

    # Proximity penalty
    for obs in obstacles:
        gap = abs(car.position - obs.position)
        if car.lane == obs.lane:
            if gap < 5:
                return -15.0, True
            elif gap < 20:
                reward -= 0.5

    if not (0 <= car.lane < road.lanes):
        return -8.0, True

    if car.position >= road.length:
        return 25.0, True

    return reward, False


def train(args=None):
    if args is None:
        args = parse_train_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    road = Road(lanes=3, length=1000)
    car = Car(lane=1, position=0.0, speed=0.0)
    sensors = [Sensor(car, angle=a, max_range=150) for a in [-90, -45, 0, 45, 90]]
    obstacles = [Obstacle(lane=random.randint(0, 2), position=random.uniform(80, 950))
                 for _ in range(12)]

    state_size = len(sensors) + 4
    action_size = 5

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=getattr(args, 'lr', 0.001),
        gamma=getattr(args, 'gamma', 0.99),
        batch_size=getattr(args, 'batch_size', 64)
    )

    history = {
        "rewards": [], "steps": [], "epsilons": [],
        "collisions": 0, "completions": 0
    }

    print("=" * 55)
    print("  SELF-DRIVING CAR — DQN TRAINING")
    print(f"  Episodes: {args.episodes} | State: {state_size}D | Actions: {action_size}")
    print("=" * 55)

    start = time.time()

    for ep in range(1, args.episodes + 1):
        car.reset()
        for obs in obstacles:
            obs.reset()

        state = build_state(sensors, car)
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done and ep_steps < 600:
            action = agent.act(state)
            car.step(action, road)
            for obs in obstacles:
                obs.step()

            reward, done = compute_reward(car, obstacles, road)
            next_state = build_state(sensors, car)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            ep_reward += reward
            ep_steps += 1

        history["rewards"].append(round(ep_reward, 2))
        history["steps"].append(ep_steps)
        history["epsilons"].append(round(agent.epsilon, 4))

        if done and ep_reward < 0:
            history["collisions"] += 1
        elif done and ep_reward > 0:
            history["completions"] += 1

        if ep % 10 == 0:
            avg_r = sum(history["rewards"][-10:]) / 10
            elapsed = time.time() - start
            print(f"  Ep {ep:4d}/{args.episodes} | "
                  f"AvgReward: {avg_r:7.2f} | "
                  f"Steps: {ep_steps:3d} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Time: {elapsed:.0f}s")

        if ep % args.save_every == 0:
            path = os.path.join(args.output_dir, f"dqn_ep{ep}.pth")
            agent.save(path)
            print(f"  [Saved] {path}")

    # Final save
    agent.save(os.path.join(args.output_dir, "dqn_final.pth"))

    log_path = os.path.join(args.log_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  Training complete in {time.time()-start:.1f}s")
    print(f"  Collisions: {history['collisions']} | Completions: {history['completions']}")
    print(f"  Log: {log_path}")
    print("=" * 55)


if __name__ == "__main__":
    train()