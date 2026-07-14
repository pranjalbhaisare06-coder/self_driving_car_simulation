"""
environment/road.py
Models the road geometry, lanes, speed limits, and traffic rules.
"""

import random


class Road:
    """
    A straight multi-lane road segment.
    
    Attributes:
        lanes      : number of lanes (default 3)
        length     : road length in position units
        speed_limit: max allowed speed in km/h
        lane_width : metres per lane (for visualisation)
    """

    LANE_COLORS = ["#3B82F6", "#10B981", "#F59E0B"]  # left, centre, right

    def __init__(self, lanes: int = 3, length: float = 1000.0,
                 speed_limit: float = 80.0):
        if lanes < 1:
            raise ValueError("Road must have at least one lane.")
        self.lanes = lanes
        self.length = length
        self.speed_limit = speed_limit
        self.lane_width = 3.7   # metres (standard highway lane)

        # Road markings — positions of hazard zones (e.g., construction)
        self.hazard_zones = self._generate_hazard_zones()

    def _generate_hazard_zones(self):
        """Randomly place 0–2 reduced-speed hazard zones."""
        zones = []
        for _ in range(random.randint(0, 2)):
            start = random.uniform(200, 700)
            zones.append({
                "start": start,
                "end": start + random.uniform(50, 150),
                "speed_limit": 40.0,
                "lane": random.randint(0, self.lanes - 1)
            })
        return zones

    def speed_limit_at(self, position: float, lane: int) -> float:
        """Return applicable speed limit at a given position and lane."""
        for zone in self.hazard_zones:
            if zone["start"] <= position <= zone["end"] and zone["lane"] == lane:
                return zone["speed_limit"]
        return self.speed_limit

    def is_valid_lane(self, lane: int) -> bool:
        return 0 <= lane < self.lanes

    def progress(self, position: float) -> float:
        """Return completion as a fraction 0–1."""
        return min(1.0, max(0.0, position / self.length))

    def get_info(self) -> dict:
        return {
            "lanes": self.lanes,
            "length": self.length,
            "speed_limit": self.speed_limit,
            "lane_width_m": self.lane_width,
            "hazard_zones": len(self.hazard_zones)
        }

    def reset(self):
        self.hazard_zones = self._generate_hazard_zones()

    def __repr__(self):
        return (f"Road(lanes={self.lanes}, length={self.length}, "
                f"limit={self.speed_limit} km/h, "
                f"hazards={len(self.hazard_zones)})")