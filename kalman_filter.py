"""Kalman Filter implementation for object tracking using FilterPy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import DT, PROCESS_NOISE_STD, MAH_DISTANCE_THRESHOLD

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKalman
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FilterPy is required for kalman_filter.KalmanFilter. "
        "Install it with `pip install filterpy`."
    ) from exc


def _build_transition_matrix(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def _build_process_noise(dt: float, q_std: float) -> np.ndarray:
    q = q_std**2
    return np.array(
        [
            [q * dt**3 / 3, 0.0, q * dt**2 / 2, 0.0],
            [0.0, q * dt**3 / 3, 0.0, q * dt**2 / 2],
            [q * dt**2 / 2, 0.0, q * dt, 0.0],
            [0.0, q * dt**2 / 2, 0.0, q * dt],
        ]
    )


def _build_measurement_matrix() -> np.ndarray:
    return np.eye(4)


class KalmanFilter:
    """Constant-velocity FilterPy-based Kalman filter."""

    def __init__(self, initial_state: np.ndarray, initial_cov: np.ndarray):
        self._kf = _FilterPyKalman(dim_x=4, dim_z=4)
        self._kf.x = initial_state.reshape(-1, 1)
        self._kf.P = initial_cov.copy()
        self._kf.F = _build_transition_matrix(DT)
        self._kf.H = _build_measurement_matrix()
        self._kf.Q = _build_process_noise(DT, PROCESS_NOISE_STD)
        # R is provided per update call

    def predict(self):
        self._kf.predict()

    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray):
        self._kf.R = measurement_cov[:4, :4]
        self._kf.update(measurement.reshape(-1, 1))

    def get_state(self) -> np.ndarray:
        return self._kf.x.reshape(-1)

    def get_position(self) -> np.ndarray:
        return self._kf.x[[0, 1], 0]

    def get_covariance(self) -> np.ndarray:
        return self._kf.P.copy()

    def mahalanobis_distance_squared(self, measurement: np.ndarray, measurement_cov: np.ndarray) -> float:
        H = self._kf.H
        S = H @ self._kf.P @ H.T + measurement_cov
        innovation = measurement - (H @ self._kf.x).reshape(-1)
        try:
            S_inv = np.linalg.inv(S)
            return float(innovation.T @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            return float("inf")


def associate_measurements_to_tracks(tracks, measurements, measurement_covs):
    """Associate measurements to tracks using Hungarian NN with χ² gating.

    Uses Hungarian algorithm for optimal one-to-one assignment with Mahalanobis gating.

    Args:
        tracks: List of KalmanFilter objects
        measurements: List of numpy arrays [x, y, vx, vy]
        measurement_covs: List of 4x4 covariance matrices

    Returns:
        associations: Dict mapping track_idx -> measurement_idx
        unassociated_measurements: List of measurement indices
    """
    if len(tracks) == 0 or len(measurements) == 0:
        return {}, list(range(len(measurements)))

    num_tracks = len(tracks)
    num_meas = len(measurements)

    # Build cost matrix using squared Mahalanobis distance
    cost_matrix = np.full((num_tracks, num_meas), np.inf)

    for i, track in enumerate(tracks):
        for j, (meas, meas_cov) in enumerate(zip(measurements, measurement_covs)):
            # Use squared Mahalanobis distance
            cost_matrix[i, j] = track.mahalanobis_distance_squared(meas, meas_cov)

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply χ² gating threshold
    associations = {}
    used_measurements = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < MAH_DISTANCE_THRESHOLD:
            associations[r] = c
            used_measurements.add(c)

    unassociated_measurements = [j for j in range(num_meas) if j not in used_measurements]

    return associations, unassociated_measurements
