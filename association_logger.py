"""Logger for Hungarian algorithm associations to detect false associations and ghost tracks."""

import csv
import os
import numpy as np
from typing import Dict, List


class AssociationLogger:
    """Logs all Hungarian algorithm associations for FA and ghost analysis."""

    def __init__(self, output_path: str = "out/association_log.csv"):
        """Initialize logger.

        Args:
            output_path: Path to output CSV file
        """
        self.output_path = output_path
        self._initialize_csv()

    def _initialize_csv(self):
        """Create CSV file with headers."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep',
                'sensor_id',
                'track_id',
                'real_object_id',
                'assigned_measurement_id',
                'measurement_real_id',
                'FA',
                'T_ghost'
            ])

    def _is_ghost_track(self, track, true_states: List, distance_threshold: float) -> bool:
        """Check if track is a ghost by comparing position to all ground truth objects.

        Args:
            track: SingleSensorTrack object
            true_states: List of ground truth object states [x, y, vx, vy]
            distance_threshold: Distance threshold for matching (meters)

        Returns:
            True if track position doesn't match any ground truth object within threshold
        """
        if true_states is None or len(true_states) == 0:
            # Can't determine without ground truth
            return False

        track_pos = track.kf.get_position()  # [x, y]

        # Calculate distance to all ground truth objects
        min_distance = float('inf')
        for true_state in true_states:
            true_pos = np.asarray(true_state)[:2]  # [x, y]
            distance = np.linalg.norm(track_pos - true_pos)
            min_distance = min(min_distance, distance)

        # Ghost if no ground truth object within threshold
        return min_distance > distance_threshold

    def log_associations(self, timestep: int, sensor_id: str,
                        associations: Dict[int, int],
                        tracks: List,
                        true_obj_ids: List,
                        true_states: List = None,
                        distance_threshold: float = 5.0):
        """Log associations for one sensor at one timestep.

        Args:
            timestep: Current timestep
            sensor_id: Sensor identifier
            associations: Dict mapping track_idx -> measurement_idx
            tracks: List of SingleSensorTrack objects
            true_obj_ids: List of ground truth object IDs for measurements
            true_states: List of ground truth object states [x, y, vx, vy]
            distance_threshold: Distance threshold for ghost detection (meters)
        """
        rows = []

        # Log associated tracks
        for track_idx, meas_idx in associations.items():
            track = tracks[track_idx]
            real_object_id = track.associated_object_id
            measurement_real_id = true_obj_ids[meas_idx]

            # Check for false association
            if real_object_id is not None and measurement_real_id != real_object_id:
                fa = f"FA(wrong={measurement_real_id}, correct={real_object_id})"
            else:
                fa = ""

            # Check if ghost track (position-based)
            is_ghost = self._is_ghost_track(track, true_states, distance_threshold)

            rows.append([
                timestep,
                sensor_id,
                track.track_id,
                real_object_id if real_object_id is not None else "",
                meas_idx,
                measurement_real_id,
                fa,
                is_ghost
            ])

        # Log unassociated tracks (potential ghosts)
        associated_track_indices = set(associations.keys())
        for track_idx, track in enumerate(tracks):
            if track_idx not in associated_track_indices:
                real_object_id = track.associated_object_id
                # Check if ghost track (position-based)
                is_ghost = self._is_ghost_track(track, true_states, distance_threshold)

                rows.append([
                    timestep,
                    sensor_id,
                    track.track_id,
                    real_object_id if real_object_id is not None else "",
                    "",  # No measurement assigned
                    "",  # No measurement real ID
                    "",  # No FA
                    is_ghost
                ])

        # Write all rows
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
