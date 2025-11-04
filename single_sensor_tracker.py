"""Single-sensor tracker with Lima's DS existence reasoning.

Each sensor maintains its own set of tracks independently.
Implements the sensor-level loop from Lima's thesis.
"""

import numpy as np
from kalman_filter import KalmanFilter, associate_measurements_to_tracks
from ds_lima import (ExistenceBBA, birth_mass, temporal_discounting,
                      update_on_association, should_delete, get_track_status)


class SingleSensorTrack:
    """Track maintained by a single sensor."""

    def __init__(self, track_id, initial_state, initial_cov, existence_bba):
        """Initialize track.

        Args:
            track_id: Unique track identifier
            initial_state: numpy array [x, y, vx, vy]
            initial_cov: 4x4 covariance matrix
            existence_bba: ExistenceBBA object
        """
        self.track_id = track_id
        self.kf = KalmanFilter(initial_state, initial_cov)
        self.existence = existence_bba  # ExistenceBBA
        self.age = 0
        self.time_since_update = 0
        self.associated_object_id = None  # For evaluation

    def predict(self):
        """Predict state and apply temporal discounting to existence."""
        self.kf.predict()
        self.existence = temporal_discounting(self.existence)
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement, measurement_cov):
        """Update with measurement."""
        self.kf.update(measurement, measurement_cov)
        self.existence = update_on_association(self.existence)
        self.time_since_update = 0

    def get_status(self):
        """Get track status."""
        return get_track_status(self.existence)


class SingleSensorTracker:
    """Single-sensor tracker implementing Lima's sensor-level existence logic."""

    def __init__(self, sensor_id, sensor_type):
        """Initialize tracker.

        Args:
            sensor_id: Sensor identifier
            sensor_type: Type of sensor ('camera', 'radar', 'lidar')
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.tracks = []
        self.next_track_id = 0

    def process_detections(self, detections):
        """Process detections for one time step (Lima's sensor-level loop)."""
        self._predict_all_tracks()

        if len(detections) == 0:
            self._delete_tracks()
            return

        measurements, measurement_covs, true_obj_ids = self._extract_detection_data(detections)
        associations, unassociated = self._associate_detections(measurements, measurement_covs)
        self._update_associated_tracks(associations, measurements, measurement_covs, true_obj_ids)
        self._birth_new_tracks(unassociated, measurements, measurement_covs, true_obj_ids)
        self._delete_tracks()

    def _predict_all_tracks(self):
        """Step 1: Predict all tracks with temporal discounting."""
        for track in self.tracks:
            track.predict()

    def _extract_detection_data(self, detections):
        """Extract measurements, covariances, and IDs from detections."""
        measurements = [d[0] for d in detections]
        measurement_covs = [d[1] for d in detections]
        true_obj_ids = [d[2] for d in detections]
        return measurements, measurement_covs, true_obj_ids

    def _associate_detections(self, measurements, measurement_covs):
        """Step 2: Associate detections to tracks using Mahalanobis distance."""
        return associate_measurements_to_tracks(
            [t.kf for t in self.tracks],
            measurements,
            measurement_covs
        )

    def _update_associated_tracks(self, associations, measurements, measurement_covs, true_obj_ids):
        """Step 3: Update tracks that associated with detections."""
        for track_idx, meas_idx in associations.items():
            track = self.tracks[track_idx]
            track.update(measurements[meas_idx], measurement_covs[meas_idx])

            if track.associated_object_id is None:
                track.associated_object_id = true_obj_ids[meas_idx]

    def _birth_new_tracks(self, unassociated, measurements, measurement_covs, true_obj_ids):
        """Step 4: Create new tracks from unmatched detections."""
        for meas_idx in unassociated:
            self._create_track(
                measurements[meas_idx],
                measurement_covs[meas_idx],
                true_obj_ids[meas_idx]
            )

    def _create_track(self, measurement, measurement_cov, true_obj_id=None):
        """Create new track from detection (birth).

        Args:
            measurement: numpy array [x, y, vx, vy]
            measurement_cov: 4x4 covariance matrix
            true_obj_id: True object ID (for evaluation)
        """
        # Initialize state from measurement
        initial_state = measurement.copy()

        # Initialize covariance with measurement covariance
        initial_cov = measurement_cov.copy()

        # Initialize existence with birth mass
        existence_bba = birth_mass()

        track = SingleSensorTrack(
            self.next_track_id,
            initial_state,
            initial_cov,
            existence_bba
        )
        track.associated_object_id = true_obj_id

        self.tracks.append(track)
        self.next_track_id += 1

    def _delete_tracks(self):
        """Delete tracks with excessive unknown mass."""
        self.tracks = [t for t in self.tracks if not should_delete(t.existence)]

    def get_track_states(self):
        """Get all track states.

        Returns:
            List of dicts containing track information
        """
        return [{
            'track_id': t.track_id,
            'state': t.kf.get_state(),
            'position': t.kf.get_position(),
            'covariance': t.kf.get_covariance(),
            'existence': {
                'm_E': t.existence.m_E,
                'm_NE': t.existence.m_NE,
                'm_U': t.existence.m_U,
                'belief': t.existence.belief(),
                'pignistic': t.existence.pignistic()
            },
            'status': t.get_status(),
            'age': t.age,
            'time_since_update': t.time_since_update,
            'associated_object_id': t.associated_object_id,
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type
        } for t in self.tracks]

    def get_num_tracks(self):
        """Get number of active tracks."""
        return len(self.tracks)

    def get_confirmed_tracks(self):
        """Get only confirmed tracks."""
        return [t for t in self.get_track_states() if t['status'] == 'confirmed']
