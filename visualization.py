"""Visualization helpers for multi-vehicle tracking system.

Includes:
1. Track counts vs ground truth plot
2. Comprehensive spatial visualization with uncertainty ellipses
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_track_counts_vs_truth(time_steps: List[int],
                               sensor_counts: Dict[str, List[int]],
                               fused_counts: List[int],
                               ground_truth_count: int,
                               save_path: str):
    """Plot confirmed track counts per sensor and fused output versus the true object count."""
    if not time_steps:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    # Ground truth reference line
    ax.plot(time_steps, [ground_truth_count] * len(time_steps),
            color="black", linestyle="--", linewidth=2,
            label="Ground truth objects")

    colour_cycle = plt.get_cmap("tab10")
    for idx, (sensor, counts) in enumerate(sensor_counts.items()):
        if not counts:
            continue
        ax.plot(time_steps, counts, linewidth=2,
                color=colour_cycle(idx),
                label=f"{sensor.capitalize()} confirmed tracks")

    if fused_counts:
        ax.plot(time_steps, fused_counts, linewidth=2,
                color="#d95f02",
                label="Fused tracks")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Track count")
    ax.set_title("Confirmed track counts vs ground truth")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_confidence_ellipse(mean: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate 95% confidence ellipse parameters using chi-squared distribution.

    Args:
        mean: 2D position [x, y]
        cov: 2x2 covariance matrix for position
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        (width, height, angle_deg): Ellipse parameters
    """
    # Chi-squared critical value for 2 DOF (x, y position)
    # For 95% confidence: χ²(2, 0.95) = 5.991
    s = chi2.ppf(confidence, df=2)

    # Eigendecomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Ellipse dimensions (eigenvalues give variance along principal axes)
    width = 2 * np.sqrt(s * eigvals[0])
    height = 2 * np.sqrt(s * eigvals[1])

    # Rotation angle
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return width, height, angle


def plot_spatial_snapshot(
    sim: Any,
    time_step: int,
    vehicle_idx: int = 0,
    save_path: str = "spatial_snapshot.png"
):
    """Plot comprehensive spatial view at a specific time step.

    Shows:
    - Vehicle position
    - Ground truth object positions
    - Predicted object positions (fused tracks)
    - Sensor measurements
    - 95% confidence uncertainty ellipses

    Args:
        sim: Simulation object with history
        time_step: Time step to visualize
        vehicle_idx: Which vehicle to show (default 0)
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get data at this time step
    vehicle_pos = sim.vehicle_history[vehicle_idx][time_step]['position']
    gt_objects = sim.object_history[time_step]
    sensor_measurements = sim.vehicle_history[vehicle_idx][time_step].get('sensor_measurements', {})
    single_sensor_tracks = sim.vehicle_history[vehicle_idx][time_step].get('single_sensor_tracks', {})
    fused_tracks = sim.vehicle_history[vehicle_idx][time_step].get('fused_tracks', [])

    # === Plot Vehicle ===
    ax.plot(vehicle_pos[0], vehicle_pos[1], 'k^', markersize=15,
            label='Vehicle', markeredgewidth=2, markerfacecolor='none')
    ax.text(vehicle_pos[0], vehicle_pos[1] + 2, f'V{vehicle_idx}',
            ha='center', fontsize=10, fontweight='bold')

    # === Plot Ground Truth Objects ===
    for obj_id, obj_state in gt_objects.items():
        ax.plot(obj_state[0], obj_state[1], 'ko', markersize=12,
                label='Ground Truth' if obj_id == 0 else '',
                markeredgewidth=2, markerfacecolor='lightgray')
        ax.text(obj_state[0] + 1, obj_state[1] + 1, f'GT{obj_id}',
                fontsize=9, color='black')

    # === Plot Sensor Measurements ===
    sensor_colors = {'camera': 'blue', 'radar': 'green', 'lidar': 'orange'}
    sensor_markers = {'camera': 'x', 'radar': 's', 'lidar': 'd'}

    for sensor_name, measurements in sensor_measurements.items():
        if not measurements:
            continue
        color = sensor_colors.get(sensor_name, 'gray')
        marker = sensor_markers.get(sensor_name, 'o')

        for i, (meas, meas_cov) in enumerate(measurements):
            ax.plot(meas[0], meas[1], marker, color=color, markersize=8,
                    alpha=0.6, markeredgewidth=1.5, markerfacecolor='none',
                    label=f'{sensor_name.capitalize()} meas' if i == 0 else '')

    # === Plot Single Sensor Tracks with Uncertainty ===
    track_colors = {'camera': 'blue', 'radar': 'green', 'lidar': 'orange'}

    for sensor_name, tracks in single_sensor_tracks.items():
        if not tracks:
            continue
        color = track_colors.get(sensor_name, 'gray')

        for i, track in enumerate(tracks):
            # Access state through kf (KalmanFilter) methods
            state = track.kf.get_state()
            cov = track.kf.get_covariance()
            pos = state[:2]
            pos_cov = cov[:2, :2]

            # Plot track position
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=10,
                    alpha=0.8, markeredgewidth=2, markerfacecolor='white',
                    label=f'{sensor_name.capitalize()} track' if i == 0 else '')

            # Plot 95% confidence ellipse
            width, height, angle = get_confidence_ellipse(pos, pos_cov, confidence=0.95)
            ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle,
                            facecolor='none', edgecolor=color, linewidth=1.5,
                            linestyle='--', alpha=0.5)
            ax.add_patch(ellipse)

    # === Plot Fused Tracks with Uncertainty ===
    for i, track in enumerate(fused_tracks):
        # Handle TrackCluster objects from multi_sensor_fusion
        if hasattr(track, 'fused_state') and track.fused_state is not None:
            pos = track.fused_state[:2]
            pos_cov = track.fused_cov[:2, :2]
        elif hasattr(track, 'kf'):
            # SingleSensorTrack - use KalmanFilter methods
            state = track.kf.get_state()
            cov = track.kf.get_covariance()
            pos = state[:2]
            pos_cov = cov[:2, :2]
        else:
            # Unknown track type, skip
            continue

        # Plot fused track position (star marker)
        ax.plot(pos[0], pos[1], '*', color='red', markersize=15,
                alpha=0.9, markeredgewidth=2, markeredgecolor='darkred',
                label='Fused track' if i == 0 else '')

        # Plot 95% confidence ellipse
        width, height, angle = get_confidence_ellipse(pos, pos_cov, confidence=0.95)
        ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle,
                        facecolor='none', edgecolor='red', linewidth=2,
                        linestyle='-', alpha=0.7)
        ax.add_patch(ellipse)

        # Label track ID
        ax.text(pos[0] + 1.5, pos[1] - 1.5, f'F{i}',
                fontsize=8, color='red', fontweight='bold')

    # === Formatting ===
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)

    # Get DT from config
    try:
        from config import DT
        time_str = f' (t={time_step*DT:.1f}s)'
    except:
        time_str = ''

    ax.set_title(f'Multi-Sensor Tracking - Time Step {time_step}{time_str}\n'
                 f'Vehicle {vehicle_idx}: {len(fused_tracks)} fused tracks, '
                 f'{len(gt_objects)} ground truth objects',
                 fontsize=14, fontweight='bold')

    ax.grid(True, linestyle=':', alpha=0.3)
    ax.axis('equal')

    # Legend with proper ordering
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='upper right', fontsize=10, framealpha=0.9)

    # Add legend for uncertainty ellipses
    ax.text(0.02, 0.98, '95% Confidence Ellipses:\n'
            '  Dashed = Single sensor\n'
            '  Solid = Fused tracks',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spatial visualization to {save_path}")
