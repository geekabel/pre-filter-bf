"""Sensor models for the multi-vehicle tracking simulation."""

import numpy as np
from config import (SENSOR_NOISE_VNC, CAMERA_AR_COEFFICIENT, RADAR_AR_COEFFICIENT,
                   LIDAR_AR_COEFFICIENT, SENSOR_TYPES)


# AR-1 coefficients per sensor type
AR_COEFFICIENTS = {
    'camera': CAMERA_AR_COEFFICIENT,
    'radar': RADAR_AR_COEFFICIENT,
    'lidar': LIDAR_AR_COEFFICIENT
}


class Sensor:
    """Base sensor class that generates measurements with full state [x, y, vx, vy]."""

    def __init__(self, sensor_type, vehicle_id, sensor_id):
        self.sensor_type = sensor_type
        self.vehicle_id = vehicle_id
        self.sensor_id = sensor_id
        self.noise_std = SENSOR_NOISE_VNC[sensor_type]  # [sigma_x, sigma_y, sigma_vx, sigma_vy]

        # AR-1 coefficient for time-correlated noise
        self.ar_coeff = AR_COEFFICIENTS[sensor_type]

        # Previous noise for AR-1 process
        self.prev_noise = np.zeros(4)  # State has 4 dimensions

    def measure(self, true_state):
        """Generate a noisy measurement of the true state.

        All sensors now use AR-1 Markov process for time-correlated noise:
        - Camera: high correlation (ρ=0.7)
        - Radar: moderate correlation (ρ=0.5)
        - Lidar: low correlation (ρ=0.3)

        Args:
            true_state: numpy array [x, y, vx, vy]

        Returns:
            measurement: numpy array [x, y, vx, vy] with added noise
            noise_cov: 4x4 covariance matrix of the measurement noise
        """
        # Time-correlated noise using AR-1 Markov process for ALL sensors
        white_noise = np.random.randn(4) * self.noise_std
        noise = self.ar_coeff * self.prev_noise + \
                np.sqrt(1 - self.ar_coeff**2) * white_noise
        self.prev_noise = noise

        measurement = true_state + noise

        # Covariance is diagonal with sensor-specific noise variances
        noise_cov = np.diag(self.noise_std**2)

        return measurement, noise_cov

    def reset_noise(self):
        """Reset the noise state (for new simulations)."""
        self.prev_noise = np.zeros(4)


class SensorArray:
    """Array of sensors for a single vehicle."""

    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.sensors = [
            Sensor(SENSOR_TYPES[0], vehicle_id, 0),  # camera
            Sensor(SENSOR_TYPES[1], vehicle_id, 1),  # radar
            Sensor(SENSOR_TYPES[2], vehicle_id, 2)   # lidar
        ]

    def measure_objects(self, object_states):
        """Generate measurements for all objects from all sensors.

        Args:
            object_states: List of numpy arrays, each [x, y, vx, vy]

        Returns:
            measurements: Dict with sensor_id as key, list of measurements as value
                         Each measurement is (state, covariance, object_idx)
        """
        measurements = {}

        for sensor in self.sensors:
            sensor_measurements = []
            for obj_idx, obj_state in enumerate(object_states):
                meas, cov = sensor.measure(obj_state)
                sensor_measurements.append((meas, cov, obj_idx))
            measurements[sensor.sensor_id] = sensor_measurements

        return measurements

    def get_sensor_type(self, sensor_id):
        """Get sensor type by sensor ID."""
        return self.sensors[sensor_id].sensor_type

    def reset(self):
        """Reset all sensors."""
        for sensor in self.sensors:
            sensor.reset_noise()
