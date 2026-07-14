"""
agent/replay_buffer.py
Experience replay buffer for DQN training.

Supports uniform random sampling. Can be extended for
Prioritised Experience Replay (PER) by overriding sample().
"""

import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Fixed-size circular buffer storing (s, a, r, s', done) transitions.
    
    Parameters:
        capacity: maximum number of transitions to store
        seed    : optional random seed for reproducibility
    """

    def __init__(self, capacity: int = 50_000, seed: int = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
        self._push_count = 0

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.buffer.append(Transition(state, action, float(reward), next_state, bool(done)))
        self._push_count += 1

    def sample(self, batch_size: int) -> list:
        """
        Sample a random mini-batch of transitions.
        Returns a list of Transition namedtuples.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested {batch_size} samples but buffer only has {len(self.buffer)}."
            )
        return random.sample(self.buffer, batch_size)

    def sample_as_arrays(self, batch_size: int):
        """
        Return five separate lists: states, actions, rewards, next_states, dones.
        Convenient for vectorised PyTorch training.
        """
        batch = self.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def clear(self):
        self.buffer.clear()
        self._push_count = 0

    @property
    def is_ready(self, batch_size: int = 64) -> bool:
        return len(self.buffer) >= batch_size

    def get_stats(self) -> dict:
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_pushed": self._push_count,
            "fill_pct": round(100 * len(self.buffer) / self.capacity, 1)
        }

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return (f"ReplayBuffer(size={len(self.buffer)}/{self.capacity}, "
                f"pushed={self._push_count})")