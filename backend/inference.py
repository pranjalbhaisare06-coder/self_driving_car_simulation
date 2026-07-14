"""
backend/inference.py
Inference module for AI Self Driving Car Simulation.
"""

from environment.car import Car
from environment.road import Road

road = Road(lanes=3, length=1000)
car = Car()


def predict(action=4):
    """
    Simulate one step of the car.

    Actions:
    0 - Accelerate
    1 - Brake
    2 - Lane Left
    3 - Lane Right
    4 - Maintain Speed
    """

    car.step(action, road)

    return {
        "lane": car.lane,
        "position": round(car.position, 2),
        "speed": round(car.speed, 2),
        "heading": round(car.heading, 2),
        "collision": car.collision,
        "total_distance": round(car.total_distance, 2)
    }


def reset():
    """Reset simulation."""

    car.reset()

    return {
        "message": "Simulation Reset",
        "state": car.get_state()
    }


def get_state():
    """Return current state."""

    return car.get_state()