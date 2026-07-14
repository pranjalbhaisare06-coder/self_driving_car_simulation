"""
environment/sensor.py
Ray-cast distance sensor attached to the autonomous car.
Returns normalised (distance, occupied) pairs.
"""

import math


class Sensor:
    """
    A single distance-sensor ray.
    
    Parameters:
        car      : the Car instance this sensor is attached to
        angle    : mounting angle in degrees (-90 left … 0 front … 90 right)
        max_range: maximum detection range in position units
    """

    def __init__(self, car, angle: float = 0.0, max_range: float = 150.0):
        self.car = car
        self.angle = angle          # degrees relative to car heading
        self.max_range = max_range

    def read(self, obstacles: list) -> tuple[float, float]:
        """
        Cast a ray and return (normalised_distance, occupied).
        
        normalised_distance: 0.0 = at object, 1.0 = nothing in range
        occupied           : 1.0 if obstacle detected, 0.0 otherwise
        """
        ray_angle = math.radians(self.car.heading + self.angle)
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)   # dy maps to lane offset

        best_dist = self.max_range
        detected = 0.0

        for obs in obstacles:
            # Project obstacle relative to car
            rel_pos = obs.position - self.car.position
            rel_lane = (obs.lane - self.car.lane) * 10.0  # lane → units

            # Simple dot-product distance along ray direction
            t = rel_pos * dx + rel_lane * dy
            if t <= 0:
                continue  # obstacle is behind

            # Perpendicular offset (width check)
            perp = abs(rel_pos * (-dy) + rel_lane * dx)
            if perp > 5.0:
                continue  # outside beam width

            if t < best_dist:
                best_dist = t
                detected = 1.0

        normalised = 1.0 - (best_dist / self.max_range)
        return normalised, detected

    def read_flat(self) -> float:
        """Return single normalised distance (for compact state vectors)."""
        dist, _ = self.read([])   # will be called with real obstacles from agent
        return dist

    def get_info(self) -> dict:
        return {
            "angle": self.angle,
            "max_range": self.max_range,
            "car_lane": self.car.lane,
            "car_pos": round(self.car.position, 1)
        }

    def __repr__(self):
        return f"Sensor(angle={self.angle}°, range={self.max_range})"