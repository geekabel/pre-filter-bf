"""Minimal CSV logging focused on pre-filter metrics."""

import csv
from pathlib import Path
from typing import Dict


class SimulationLogger:
    """Logs only the metrics required to assess the toy pre-filter."""

    def __init__(self, output_dir="out"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.fusion_metrics_log = []
        self.sensor_summary_log = []

    def log_metrics(self, timestep: int, time_sec: float, metrics: Dict[str, float]):
        """Log fleet-level fused metrics for this timestep."""
        record = {
            'timestep': timestep,
            'time_sec': time_sec,
            'avg_tracks': metrics.get('num_tracks', 0.0),
            'avg_ghost_tracks': metrics.get('ghost_tracks', 0.0),
            'avg_missed_objects': metrics.get('missed_objects', 0.0),
            'avg_false_associations': metrics.get('false_associations', 0.0),
            'mean_position_error': metrics.get('tracking_error', 0.0),
            'rmse_position_error': metrics.get('rmse_tracking_error', 0.0),
            'vehicles_with_four_tracks': metrics.get('vehicles_with_four_tracks', 0),
            'perfect_vehicles': metrics.get('perfect_vehicles', 0)
        }
        self.fusion_metrics_log.append(record)

    def log_sensor_summary(self, timestep: int, time_sec: float, vehicle_id: int,
                           sensor_type: str, total_tracks: int,
                           confirmed_tracks: int, fused_tracks: int):
        """Log per-sensor track counts to show existence gating impact."""
        record = {
            'timestep': timestep,
            'time_sec': time_sec,
            'vehicle_id': vehicle_id,
            'sensor_type': sensor_type,
            'total_sensor_tracks': total_tracks,
            'confirmed_sensor_tracks': confirmed_tracks,
            'fused_tracks_vehicle': fused_tracks
        }
        self.sensor_summary_log.append(record)

    def save_all(self):
        """Persist the two CSV files with relevant metrics."""
        self._save_csv('fusion_metrics.csv', self.fusion_metrics_log)
        self._save_csv('sensor_existence_summary.csv', self.sensor_summary_log)

        print(f"\nLogs saved to {self.output_dir}/")

    def _save_csv(self, filename, data):
        if len(data) == 0:
            return

        filepath = self.output_dir / filename
        with filepath.open('w', newline='') as fh:
            writer = csv.DictWriter(fh , fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
