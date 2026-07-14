"""
Self-Driving Car Simulation - Main Entry Point
Launches the simulation environment with a trained or untrained DQN agent.
"""

import os
import argparse
import json
import time
import random
import math
from environment.car import Car
from environment.road import Road
from environment.sensor import Sensor
from environment.obstacle import Obstacle
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Driving Car Simulation")
    parser.add_argument("--mode", choices=["train", "simulate", "demo"], default="demo",
                        help="Run mode: train, simulate, or demo")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--render", action="store_true", help="Render simulation")
    parser.add_argument("--log", type=str, default="logs/run_log.json", help="Log output path")
    return parser.parse_args()


def run_simulation(args):
    """Run a full simulation episode."""
    road = Road(lanes=3, length=1000)
    car = Car(lane=1, position=0.0, speed=0.0)
    sensors = [Sensor(car, angle=a, max_range=150) for a in [-45, 0, 45, -90, 90]]
    obstacles = [Obstacle(lane=random.randint(0, 2), position=random.uniform(100, 900)) 
                 for _ in range(10)]

    state_size = len(sensors) * 2 + 4  # sensor readings + car state
    action_size = 5  # accelerate, brake, left, right, maintain
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    if args.model and os.path.exists(args.model):
        agent.load(args.model)
        print(f"[INFO] Loaded model from {args.model}")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_data = {"episodes": [], "config": {"lanes": 3, "obstacles": 10}}

    print(f"[INFO] Starting simulation — mode={args.mode}, episodes={args.episodes}")

    for episode in range(args.episodes):
        car.reset()
        for obs in obstacles:
            obs.reset()
        
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:
            # Build state vector
            sensor_readings = [s.read(obstacles) for s in sensors]
            flat_sensors = [v for pair in sensor_readings for v in pair]
            state = flat_sensors + [car.speed, car.lane, car.position / 1000.0, car.heading]

            action = agent.act(state)
            car.step(action, road)
            for obs in obstacles:
                obs.step()

            reward, done = compute_reward(car, obstacles, road)
            next_sensor_readings = [s.read(obstacles) for s in sensors]
            flat_next = [v for pair in next_sensor_readings for v in pair]
            next_state = flat_next + [car.speed, car.lane, car.position / 1000.0, car.heading]

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            total_reward += reward
            steps += 1

        log_data["episodes"].append({
            "episode": episode + 1,
            "steps": steps,
            "total_reward": round(total_reward, 2),
            "epsilon": round(agent.epsilon, 4),
            "collision": done and total_reward < 0,
            "distance": round(car.position, 1)
        })

        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode+1}/{args.episodes} | "
                  f"Reward: {total_reward:.1f} | Steps: {steps} | ε: {agent.epsilon:.3f}")

    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_final.pth")
    with open(args.log, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n[DONE] Log saved to {args.log}")
    print(f"[DONE] Model saved to models/dqn_final.pth")


def compute_reward(car, obstacles, road):
    """Compute reward signal for the RL agent."""
    reward = 0.1  # survival reward

    # Speed reward
    if 30 <= car.speed <= 80:
        reward += 0.3
    elif car.speed > 80:
        reward -= 0.2

    # Lane-keeping reward
    if car.lane in [0, 1, 2]:
        reward += 0.1

    # Collision penalty
    for obs in obstacles:
        dist = abs(car.position - obs.position)
        if dist < 5 and car.lane == obs.lane:
            return -10.0, True  # collision

    # Road boundary check
    if car.lane < 0 or car.lane >= road.lanes:
        return -5.0, True

    # Goal reached
    if car.position >= road.length:
        return 20.0, True

    return reward, False


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        from train import train
        train(args)
    else:
        run_simulation(args)