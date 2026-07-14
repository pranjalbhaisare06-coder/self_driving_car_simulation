from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

# Project root ko Python path me add karo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.car import Car
from environment.road import Road
from backend.inference import predict, reset, get_state

app = Flask(__name__)
CORS(app)

road = Road(lanes=3, length=1000)
car = Car()

simulation = {
    "status": "Stopped",
    "reward": 0,
    "episode": 0,
    "speed": 0,
    "collision": False
}

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "AI Self-Driving Car Simulation API is Running"
    })
# -----------------------------
# Start Simulation
# -----------------------------
@app.route("/start")
def start():

    simulation["status"] = "Running"

    # Car reset
    car.reset()

    # Initial values
    simulation["episode"] += 1
    simulation["reward"] = 0
    simulation["collision"] = False
    simulation["speed"] = car.speed

    return jsonify({
        "message": "Simulation Started",
        "status": simulation["status"],
        "episode": simulation["episode"],
        "speed": simulation["speed"]
    })
# -----------------------------
# State Simulation
# -----------------------------
@app.route("/state")
def state():

    if simulation["status"] == "Running":
        car.step(4, road)          # Maintain Speed

    data = car.get_state()

    simulation["speed"] = data["speed"]

    data["status"] = simulation["status"]
    data["episode"] = simulation["episode"]
    data["reward"] = simulation["reward"]

    return jsonify(data)
# -----------------------------
# Stop Simulation
# -----------------------------
@app.route("/stop")
def stop():
    simulation["status"] = "Stopped"
    simulation["speed"] = 0
    return jsonify(simulation)

# -----------------------------
# Reset Simulation
# -----------------------------
@app.route("/reset")
def reset():

    simulation["status"] = "Stopped"
    simulation["episode"] = 0
    simulation["speed"] = 0
    simulation["reward"] = 0
    simulation["collision"] = False
    simulation["lane"] = "Center"

    return jsonify(simulation)

# -----------------------------
# Reward
# -----------------------------
@app.route("/reward")
def reward():
    return jsonify({
        "reward": simulation["reward"]
    })
    
# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)