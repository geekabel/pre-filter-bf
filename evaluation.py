"""Evaluation utilities for the multi-vehicle Lima pre-filter experiment.

- Each vehicle should ideally maintain exactly 4 fused tracks (one per object).
- We track ghost tracks (extra tracks), missed objects, and mis-associations.
- Metrics are computed per vehicle and aggregated across the fleet.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    track_idx: int
    truth_idx: int
    distance: float
    mis_association: bool
    uncertain: bool


@dataclass
class VehicleMetrics:
    vehicle_id: int
    num_tracks: int
    ghost_tracks: int
    missed_objects: int
    matched_objects: int
    false_associations: int
    uncertain_associations: int
    mean_position_error: float
    rmse_position_error: float
    avg_belief: float
    sensor_track_counts: Dict[str, int] = field(default_factory=dict)
    confirmed_track_counts: Dict[str, int] = field(default_factory=dict)


class TrackingEvaluator:
    """Evaluates the toy pre-filter scenario across vehicles and time."""

    def __init__(self, num_true_objects: int, assignment_threshold: float = 5.0):
        self.num_true_objects = num_true_objects
        self.assignment_threshold = assignment_threshold
        self.history = {
            'time_steps': [],
            'overall_metrics': [],
            'per_vehicle': []
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_timestep(self, time_step: int, true_states: List[np.ndarray],
                           vehicle_agents: List):
        """Evaluate every vehicle at the current timestep."""
        vehicle_results = []
        for vehicle_id, vehicle in enumerate(vehicle_agents):
            vehicle_results.append(
                self._evaluate_vehicle(vehicle_id, vehicle, true_states)
            )

        overall = self._aggregate_vehicle_metrics(vehicle_results)

        self.history['time_steps'].append(time_step)
        self.history['per_vehicle'].append(vehicle_results)
        self.history['overall_metrics'].append(overall)

    def get_history(self):
        return self.history

    def get_summary_statistics(self) -> Dict[str, float]:
        """Return fleet-level summary statistics over the whole run."""
        if not self.history['overall_metrics']:
            return {}

        overall_array = self.history['overall_metrics']
        per_vehicle_records = [
            metrics
            for timestep_record in self.history['per_vehicle']
            for metrics in timestep_record
        ]

        def avg(key):
            return float(np.mean([rec[key] for rec in overall_array]))

        def avg_vehicle(key):
            return float(np.mean([rec.__dict__[key] for rec in per_vehicle_records]))

        perfect_counts = sum(
            1 for rec in per_vehicle_records
            if rec.num_tracks == self.num_true_objects
            and rec.ghost_tracks == 0
            and rec.false_associations == 0
            and rec.missed_objects == 0
        )

        summary = {
            'avg_tracks_per_vehicle': avg_vehicle('num_tracks'),
            'avg_ghost_tracks_per_vehicle': avg_vehicle('ghost_tracks'),
            'avg_missed_objects_per_vehicle': avg_vehicle('missed_objects'),
            'avg_false_associations_per_vehicle': avg_vehicle('false_associations'),
            'fleet_mean_position_error': avg('tracking_error'),
            'fleet_rmse_position_error': avg('rmse_tracking_error'),
            'fleet_avg_belief': avg_vehicle('avg_belief'),
            'perfect_vehicle_ratio': perfect_counts / max(len(per_vehicle_records), 1),
            'total_false_associations': sum(rec.false_associations for rec in per_vehicle_records)
        }
        return summary

    def print_summary(self):
        if not self.history['per_vehicle']:
            print("No evaluation data collected yet.")
            return

        final_metrics = self.history['per_vehicle'][-1]
        print("\n=== Final Step Metrics ===")
        for vm in final_metrics:
            print(
                f"Vehicle {vm.vehicle_id}: "
                f"fused_tracks={vm.num_tracks}, "
                f"ghost_tracks={vm.ghost_tracks}, "
                f"false_associations={vm.false_associations}, "
                f"rmse_position_error={vm.rmse_position_error:.3f} m"
            )
        print("=================================\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_vehicle(self, vehicle_id: int, vehicle, true_states: List[np.ndarray]) -> VehicleMetrics:
        fused_tracks = vehicle.get_fused_tracks()

        matches, unmatched_tracks, unmatched_truth = self._associate_tracks(
            true_states, fused_tracks
        )

        false_associations = sum(1 for match in matches if match.mis_association)
        uncertain_associations = sum(1 for match in matches if match.uncertain)
        errors = [match.distance for match in matches]

        mean_error = float(np.mean(errors)) if errors else 0.0
        rmse_error = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

        beliefs = [
            track['existence']['belief']
            for track in fused_tracks
            if track.get('existence')
        ]
        avg_belief = float(np.mean(beliefs)) if beliefs else 0.0

        num_tracks_per_sensor = vehicle.get_num_tracks_per_sensor()
        confirmed_per_sensor = vehicle.get_statistics()['confirmed_per_sensor']

        return VehicleMetrics(
            vehicle_id=vehicle_id,
            num_tracks=len(fused_tracks),
            ghost_tracks=len(unmatched_tracks),
            missed_objects=len(unmatched_truth),
            matched_objects=len(matches),
            false_associations=false_associations,
            uncertain_associations=uncertain_associations,
            mean_position_error=mean_error,
            rmse_position_error=rmse_error,
            avg_belief=avg_belief,
            sensor_track_counts=num_tracks_per_sensor,
            confirmed_track_counts=confirmed_per_sensor
        )

    def _associate_tracks(self, true_states: List[np.ndarray], fused_tracks: List[dict]) \
            -> Tuple[List[MatchResult], List[int], List[int]]:
        """Assign fused tracks to truth objects using Mah distance + gating."""
        if len(true_states) == 0 or len(fused_tracks) == 0:
            return [], list(range(len(fused_tracks))), list(range(len(true_states)))

        true_positions = np.asarray([state[:2] for state in true_states])
        track_positions = np.asarray([track['position'] for track in fused_tracks])

        cost_matrix = np.linalg.norm(track_positions[:, None, :] - true_positions[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[MatchResult] = []
        used_tracks = set()
        used_truth = set()

        for track_idx, truth_idx in zip(row_ind, col_ind):
            distance = float(cost_matrix[track_idx, truth_idx])
            if distance > self.assignment_threshold:
                continue

            mis_association, uncertain = self._check_mis_association(
                fused_tracks[track_idx], truth_idx
            )

            matched_pairs.append(MatchResult(
                track_idx=track_idx,
                truth_idx=truth_idx,
                distance=distance,
                mis_association=mis_association,
                uncertain=uncertain
            ))
            used_tracks.add(track_idx)
            used_truth.add(truth_idx)

        unmatched_tracks = [idx for idx in range(len(fused_tracks)) if idx not in used_tracks]
        unmatched_truth = [idx for idx in range(len(true_states)) if idx not in used_truth]

        return matched_pairs, unmatched_tracks, unmatched_truth

    def _check_mis_association(self, fused_track: dict, truth_idx: int) -> Tuple[bool, bool]:
        """Inspect contributing sensor tracks to flag false or uncertain associations."""
        source_tracks = fused_track.get('source_tracks', [])
        associated_ids = [
            src.get('associated_object_id')
            for src in source_tracks
            if src.get('associated_object_id') is not None
        ]

        if not associated_ids:
            return False, True  # No provenance to confirm; mark as uncertain

        mis_assoc = any(obj_id != truth_idx for obj_id in associated_ids)
        return mis_assoc, False

    def _aggregate_vehicle_metrics(self, vehicle_results: List[VehicleMetrics]) -> Dict[str, float]:
        """Compute fleet-level aggregates for logging/visualisation."""
        if not vehicle_results:
            return {
                'num_tracks': 0.0,
                'ghost_tracks': 0.0,
                'missed_objects': 0.0,
                'false_associations': 0.0,
                'tracking_error': 0.0,
                'rmse_tracking_error': 0.0,
                'vehicles_with_four_tracks': 0,
                'perfect_vehicles': 0
            }

        num_tracks = np.mean([res.num_tracks for res in vehicle_results])
        ghost_tracks = np.mean([res.ghost_tracks for res in vehicle_results])
        missed_objects = np.mean([res.missed_objects for res in vehicle_results])
        false_associations = np.mean([res.false_associations for res in vehicle_results])
        mean_error = np.mean([res.mean_position_error for res in vehicle_results])
        rmse_error = np.sqrt(np.mean([res.rmse_position_error**2 for res in vehicle_results]))

        vehicles_with_four = sum(1 for res in vehicle_results if res.num_tracks == self.num_true_objects)
        perfect_vehicles = sum(
            1 for res in vehicle_results
            if res.num_tracks == self.num_true_objects
            and res.ghost_tracks == 0
            and res.false_associations == 0
            and res.missed_objects == 0
        )

        return {
            'num_tracks': float(num_tracks),
            'ghost_tracks': float(ghost_tracks),
            'missed_objects': float(missed_objects),
            'false_associations': float(false_associations),
            'tracking_error': float(mean_error),
            'rmse_tracking_error': float(rmse_error),
            'vehicles_with_four_tracks': vehicles_with_four,
            'perfect_vehicles': perfect_vehicles
        }
