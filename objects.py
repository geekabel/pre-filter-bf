"""Object dynamics for the simulation."""

import numpy as np
from config import DT, PROCESS_NOISE_STD


def add_line(initial_state, acceleration, duration=None, max_speed=None, dt=DT):
    """Add a straight-line trajectory segment.

    Args:
        initial_state: numpy array [x, y, vx, vy]
        acceleration: acceleration magnitude in m/s^2
        duration: maximum duration in seconds (None = unlimited)
        max_speed: maximum speed in m/s (None = unlimited)
        dt: time step in seconds

    Returns:
        List of states, each as numpy array [x, y, vx, vy]
    """
    trajectory = []
    x, y, vx, vy = initial_state

    # Calculate heading from initial velocity
    heading = np.arctan2(vy, vx)

    time = 0
    while True:
        # Add current state to trajectory
        trajectory.append(np.array([x, y, vx, vy]))

        # Apply acceleration in the current heading direction
        ax = acceleration * np.cos(heading)
        ay = acceleration * np.sin(heading)
        vx += ax * dt
        vy += ay * dt

        # Check speed limit
        speed = np.sqrt(vx**2 + vy**2)
        if max_speed is not None and speed >= max_speed:
            break

        # Update position
        x += vx * dt
        y += vy * dt

        # Update heading from velocity
        heading = np.arctan2(vy, vx)

        time += dt
        if duration is not None and time >= duration:
            break

    return trajectory


def add_arc(initial_state, duration=None, radius=None, angle_limit=None, dt=DT):
    """Add an arc trajectory segment at constant speed.

    Args:
        initial_state: numpy array [x, y, vx, vy]
        duration: maximum duration in seconds (None = unlimited)
        radius: turn radius in meters (None = unlimited, will go straight)
        angle_limit: angle to turn through in radians (None = unlimited)
        dt: time step in seconds

    Returns:
        List of states, each as numpy array [x, y, vx, vy]
    """
    trajectory = []
    x, y, vx, vy = initial_state

    # Calculate speed (constant throughout arc)
    speed = np.sqrt(vx**2 + vy**2)

    # Calculate heading from velocity
    heading = np.arctan2(vy, vx)

    # Calculate angular velocity (positive = left turn, negative = right turn)
    if radius is None or radius == 0:
        angular_velocity = 0
    else:
        angular_velocity = speed / radius

    time = 0
    heading_change = 0

    while True:
        # Add current state to trajectory
        trajectory.append(np.array([x, y, vx, vy]))

        # Update heading
        heading += angular_velocity * dt
        heading_change += angular_velocity * dt

        # Update position
        x += speed * dt * np.cos(heading)
        y += speed * dt * np.sin(heading)

        # Update velocity components
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        # Check angle limit
        if angle_limit is not None and abs(heading_change) >= angle_limit:
            break

        time += dt
        if duration is not None and time >= duration:
            break

    return trajectory


class MovingObject:
    """Represents a moving object with constant velocity model."""

    def __init__(self, obj_id, initial_state, trajectory=None):
        """Initialize object.

        Args:
            obj_id: Unique object identifier
            initial_state: numpy array [x, y, vx, vy]
            trajectory: Optional trajectory (list of states)
        """
        self.obj_id = obj_id
        self.state = initial_state.copy()  # [x, y, vx, vy]
        self.trajectory = trajectory
        self.trajectory_index = 0

    def update(self):
        """Update object state using constant velocity model with process noise."""
        if self.trajectory is not None and self.trajectory_index < len(self.trajectory):
            # Use trajectory
            self.state = self.trajectory[self.trajectory_index].copy()
            self.trajectory_index += 1
        else:
            # Fall back to constant velocity model
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

    def __init__(self, trajectory_specs=None, initial_states=None):
        """Initialize simulator with objects.

        Args:
            trajectory_specs: List of dicts with 'initial_state' and 'maneuvers'
            initial_states: numpy array of shape (num_objects, 4) - legacy mode
        """
        if trajectory_specs is not None:
            # New trajectory-based mode
            self.objects = []
            for i, spec in enumerate(trajectory_specs):
                initial_state = np.array(spec['initial_state'])
                trajectory = self._generate_trajectory(initial_state, spec.get('maneuvers', []))
                self.objects.append(MovingObject(i, initial_state, trajectory))
        elif initial_states is not None:
            # Legacy constant velocity mode
            self.objects = [MovingObject(i, initial_states[i])
                           for i in range(len(initial_states))]
        else:
            raise ValueError("Must provide either trajectory_specs or initial_states")

    def _generate_trajectory(self, initial_state, maneuvers):
        """Generate complete trajectory from maneuvers."""
        if not maneuvers:
            return None

        trajectory = []
        current_state = initial_state.copy()

        for m in maneuvers:
            if m['type'] == 'line':
                segment = add_line(current_state, m['acceleration'], m.get('duration'), m.get('max_speed'))
            elif m['type'] == 'arc':
                segment = add_arc(current_state, m.get('duration'), m.get('radius'), m.get('angle_limit'))
            else:
                continue

            if trajectory:
                segment = segment[1:]
            trajectory.extend(segment)
            current_state = trajectory[-1].copy()

        return trajectory

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
