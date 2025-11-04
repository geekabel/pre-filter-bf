"""Object dynamics for the simulation."""

import numpy as np
from config import DT, PROCESS_NOISE_STD


class MovingObject:
    """Represents a moving object with constant velocity model."""

    def __init__(self, obj_id, initial_state):
        """Initialize object.

        Args:
            obj_id: Unique object identifier
            initial_state: numpy array [x, y, vx, vy]
        """
        self.obj_id = obj_id
        self.state = initial_state.copy()  # [x, y, vx, vy]

    def update(self):
        """Update object state using constant velocity model with process noise."""
        # State transition: x_k+1 = x_k + vx*dt, y_k+1 = y_k + vy*dt
        self.state[0] += self.state[2] * DT  # x += vx * dt
        self.state[1] += self.state[3] * DT  # y += vy * dt

        # Add process noise to velocities
        process_noise = np.random.randn(4) * PROCESS_NOISE_STD
        self.state += process_noise

    def get_position(self):
        """Get current position [x, y]."""
        return self.state[:2].copy()

    def get_state(self):
        """Get full state [x, y, vx, vy]."""
        return self.state.copy()


class ObjectSimulator:
    """Simulates multiple objects over time."""

    def __init__(self, initial_states):
        """Initialize simulator with objects.

        Args:
            initial_states: numpy array of shape (num_objects, 4)
                           Each row is [x, y, vx, vy]
        """
        self.objects = [MovingObject(i, initial_states[i])
                       for i in range(len(initial_states))]

    def step(self):
        """Advance simulation by one time step."""
        for obj in self.objects:
            obj.update()

    def get_positions(self):
        """Get all object positions at current time.

        Returns:
            List of numpy arrays [x, y]
        """
        return [obj.get_position() for obj in self.objects]

    def get_states(self):
        """Get all object states at current time.

        Returns:
            List of numpy arrays [x, y, vx, vy]
        """
        return [obj.get_state() for obj in self.objects]
