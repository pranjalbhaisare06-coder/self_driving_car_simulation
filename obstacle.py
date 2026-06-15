"""
environment/obstacle.py
Simulates static and moving traffic obstacles on the road.
"""

import random


class Obstacle:
    """
    A road obstacle (parked car, slower vehicle, road debris).
    
    Parameters:
        lane    : lane index (0 = leftmost)
        position: initial longitudinal position
        moving  : whether this obstacle moves (traffic vehicle)
        speed   : speed if moving (km/h)
    """

    TYPES = ["vehicle", "debris", "cone", "truck"]

    def __init__(self, lane: int = 0, position: float = 100.0,
                 moving: bool = True, speed: float = None):
        self._init_lane = lane
        self._init_position = position
        self.moving = moving

        self.lane = lane
        self.position = position
        self.speed = speed if speed is not None else random.uniform(20.0, 60.0)
        self.obstacle_type = random.choice(self.TYPES)
        self.active = True

        # Occasionally change lanes (traffic behaviour)
        self._lane_change_timer = random.randint(50, 200)
        self._max_lane = 2  # set by environment

    def reset(self):
        self.lane = self._init_lane
        self.position = self._init_position + random.uniform(-20, 20)
        self.speed = random.uniform(20.0, 60.0)
        self.active = True
        self._lane_change_timer = random.randint(50, 200)

    def step(self):
        """Advance obstacle one timestep."""
        if not self.active:
            return

        if self.moving:
            dt = 1.0 / 30.0
            self.position += self.speed * dt

        # Lane-change behaviour for traffic vehicles
        if self.obstacle_type == "vehicle":
            self._lane_change_timer -= 1
            if self._lane_change_timer <= 0:
                direction = random.choice([-1, 0, 1])
                new_lane = self.lane + direction
                if 0 <= new_lane <= self._max_lane:
                    self.lane = new_lane
                self._lane_change_timer = random.randint(50, 200)

        # Wrap around road (loop traffic)
        if self.position > 1000:
            self.position = random.uniform(0, 50)

    def distance_to(self, car) -> float:
        """Euclidean-ish distance to a Car instance."""
        lane_diff = (self.lane - car.lane) * 10.0
        pos_diff = self.position - car.position
        return (pos_diff**2 + lane_diff**2) ** 0.5

    def get_info(self) -> dict:
        return {
            "type": self.obstacle_type,
            "lane": self.lane,
            "position": round(self.position, 1),
            "speed": round(self.speed, 1),
            "moving": self.moving,
            "active": self.active
        }

    def __repr__(self):
        return (f"Obstacle({self.obstacle_type}, lane={self.lane}, "
                f"pos={self.position:.1f}, speed={self.speed:.1f})")