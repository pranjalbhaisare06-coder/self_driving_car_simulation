"""
environment/car.py
Simulates the autonomous vehicle's physics and state.
"""

import math
import random


class Car:
    """
    Represents the autonomous vehicle in the simulation.
    
    State: position, speed, lane, heading
    Actions:
        0 - Accelerate
        1 - Brake
        2 - Lane Left
        3 - Lane Right
        4 - Maintain Speed
    """

    ACTION_NAMES = {
        0: "Accelerate",
        1: "Brake",
        2: "Lane Left",
        3: "Lane Right",
        4: "Maintain"
    }

    MAX_SPEED = 120.0   # km/h
    MIN_SPEED = 0.0
    ACCEL = 5.0         # speed change per step
    LANE_CHANGE_TIME = 3  # steps to complete lane change

    def __init__(self, lane: int = 1, position: float = 0.0, speed: float = 30.0):
        self._init_lane = lane
        self._init_position = position
        self._init_speed = speed

        self.lane = lane
        self.position = position
        self.speed = speed
        self.heading = 0.0          # degrees, 0 = straight ahead
        self.target_lane = lane
        self.lane_change_progress = 0
        self.total_distance = 0.0
        self.step_count = 0
        self.collision = False
        self.action_history = []

    def reset(self):
        self.lane = self._init_lane
        self.position = self._init_position
        self.speed = self._init_speed + random.uniform(-5, 5)
        self.heading = 0.0
        self.target_lane = self._init_lane
        self.lane_change_progress = 0
        self.total_distance = 0.0
        self.step_count = 0
        self.collision = False
        self.action_history = []

    def step(self, action: int, road):
        """Apply action and advance physics one timestep."""
        self.step_count += 1
        self.action_history.append(action)

        if action == 0:  # Accelerate
            self.speed = min(self.MAX_SPEED, self.speed + self.ACCEL)
        elif action == 1:  # Brake
            self.speed = max(self.MIN_SPEED, self.speed - self.ACCEL * 1.5)
        elif action == 2:  # Lane Left
            if self.lane > 0 and self.lane_change_progress == 0:
                self.target_lane = self.lane - 1
                self.lane_change_progress = self.LANE_CHANGE_TIME
        elif action == 3:  # Lane Right
            if self.lane < road.lanes - 1 and self.lane_change_progress == 0:
                self.target_lane = self.lane + 1
                self.lane_change_progress = self.LANE_CHANGE_TIME
        # action == 4: maintain

        # Lane change animation
        if self.lane_change_progress > 0:
            self.lane_change_progress -= 1
            self.heading = (self.target_lane - self.lane) * 5.0
            if self.lane_change_progress == 0:
                self.lane = self.target_lane
                self.heading = 0.0
        else:
            self.heading *= 0.8  # damp residual heading

        # Advance position (speed in km/h → position units per step)
        dt = 1.0 / 30.0  # 30 steps per second
        self.position += self.speed * dt
        self.total_distance += self.speed * dt

    def get_state(self) -> dict:
        return {
            "lane": self.lane,
            "position": round(self.position, 2),
            "speed": round(self.speed, 2),
            "heading": round(self.heading, 2),
            "target_lane": self.target_lane,
            "collision": self.collision,
            "step": self.step_count,
            "total_distance": round(self.total_distance, 1)
        }

    def __repr__(self):
        return (f"Car(lane={self.lane}, pos={self.position:.1f}, "
                f"speed={self.speed:.1f}, heading={self.heading:.1f}°)")