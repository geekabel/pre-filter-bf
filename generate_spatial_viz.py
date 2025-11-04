"""Generate spatial visualization animations from simulation run."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from config import (NUM_VEHICLES, NUM_TIME_STEPS, OBJECT_INITIAL_STATES,
                   NUM_OBJECTS, SENSOR_TYPES, DT)
from objects import ObjectSimulator
from vehicle_agent import VehicleAgent
from visualization import get_confidence_ellipse


class SimulationWithHistory:
    """Wrapper to run simulation and store history for visualization."""

    def __init__(self):
        self.object_history = []  # [{obj_id: state, ...}, ...]
        self.vehicle_history = []  # [[{step data}, ...] for each vehicle]
        self.vehicles = []
        self.object_sim = None

    def run(self):
        """Run simulation and store all history."""
        print("\n" + "="*70)
        print("Running Simulation with History Tracking")
        print("="*70 + "\n")

        # Initialize
        self.object_sim = ObjectSimulator(OBJECT_INITIAL_STATES)
        self.vehicles = [VehicleAgent(i) for i in range(NUM_VEHICLES)]
        self.vehicle_history = [[] for _ in range(NUM_VEHICLES)]

        print(f"Initialized {NUM_OBJECTS} objects, {NUM_VEHICLES} vehicles")
        print(f"Sensors per vehicle: {', '.join(SENSOR_TYPES)}")
        print(f"\nRunning {NUM_TIME_STEPS} time steps...")

        # Main loop
        for t in range(NUM_TIME_STEPS):
            # Get true object states
            true_states = self.object_sim.get_states()

            # Store object history
            self.object_history.append({i: state.copy() for i, state in enumerate(true_states)})

            # Process each vehicle
            for v_idx, vehicle in enumerate(self.vehicles):
                # Get sensor measurements BEFORE processing
                sensor_measurements = {}
                for sensor in vehicle.sensor_array.sensors:
                    measurements = []
                    for obj_state in true_states:
                        meas, meas_cov = sensor.measure(obj_state)
                        measurements.append((meas, meas_cov))
                    sensor_measurements[sensor.sensor_type] = measurements

                # Process timestep
                fused_tracks = vehicle.process_timestep(true_states)

                # Get single sensor tracks
                single_sensor_tracks = {}
                for sensor_id, sensor_type in enumerate(SENSOR_TYPES):
                    tracker = vehicle.single_sensor_trackers[sensor_id]
                    single_sensor_tracks[sensor_type] = tracker.tracks.copy()

                # Vehicle position (assume at origin for now - can be updated if vehicle moves)
                vehicle_pos = np.array([0.0, 0.0])

                # Store vehicle history
                step_data = {
                    'position': vehicle_pos.copy(),
                    'sensor_measurements': sensor_measurements,
                    'single_sensor_tracks': single_sensor_tracks,
                    'fused_tracks': fused_tracks.copy() if fused_tracks else []
                }
                self.vehicle_history[v_idx].append(step_data)

            # Update objects
            self.object_sim.step()

            if (t + 1) % 10 == 0:
                print(f"  Step {t+1:3d}/{NUM_TIME_STEPS}")

        print("\nSimulation complete!")
        return self

    def generate_animation(self, vehicle_idx=0, save_path=None):
        """Generate animation for a specific vehicle showing all timesteps.

        Args:
            vehicle_idx: Which vehicle to animate (0-3)
            save_path: Output filename (default: vehicle_{idx}_tracking.mp4)
        """
        if save_path is None:
            save_path = f"vehicle_{vehicle_idx}_tracking.mp4"

        print(f"Creating animation for Vehicle {vehicle_idx}...")
        print(f"Total frames: {len(self.object_history)}")
        print(f"Duration: {len(self.object_history) * DT:.1f}s")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        def init():
            """Initialize animation."""
            ax.clear()
            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.3)
            return []

        def update(frame):
            """Update function for animation."""
            ax.clear()

            # Get data at this time step
            vehicle_pos = self.vehicle_history[vehicle_idx][frame]['position']
            gt_objects = self.object_history[frame]
            sensor_measurements = self.vehicle_history[vehicle_idx][frame].get('sensor_measurements', {})
            single_sensor_tracks = self.vehicle_history[vehicle_idx][frame].get('single_sensor_tracks', {})
            fused_tracks = self.vehicle_history[vehicle_idx][frame].get('fused_tracks', [])

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
            # First pass: identify ghost tracks and mis-associations
            obj_to_track_indices = {}  # Map true object ID -> list of track indices
            ghost_track_indices = []
            mis_assoc_track_indices = []

            for i, track in enumerate(fused_tracks):
                if isinstance(track, dict):
                    # Dictionary format from get_fused_tracks()
                    source_tracks = track.get('source_tracks', [])
                    diagnostics = track.get('diagnostics', {})

                    # Check for ghost track (no associated object IDs)
                    is_ghost = diagnostics.get('is_ghost', False)

                    # Check for mis-association (multiple different object IDs)
                    is_mis_assoc = diagnostics.get('is_mis_association', False)

                    # Get consensus object ID
                    obj_ids = [s.get('associated_object_id') for s in source_tracks
                              if s.get('associated_object_id') is not None]

                    if is_ghost:
                        ghost_track_indices.append(i)
                    elif is_mis_assoc:
                        mis_assoc_track_indices.append(i)
                    elif obj_ids:
                        # Track with valid object association
                        consensus_id = diagnostics.get('consensus_object_id')
                        if consensus_id is not None:
                            if consensus_id not in obj_to_track_indices:
                                obj_to_track_indices[consensus_id] = []
                            obj_to_track_indices[consensus_id].append(i)

            # Identify ghost tracks: multiple tracks for same object
            for obj_id, track_indices in obj_to_track_indices.items():
                if len(track_indices) > 1:
                    # Keep the first track as "correct", mark others as ghosts
                    for idx in track_indices[1:]:
                        if idx not in ghost_track_indices:
                            ghost_track_indices.append(idx)

            # Second pass: plot tracks with appropriate colors
            good_track_plotted = False
            ghost_track_plotted = False
            mis_assoc_plotted = False

            for i, track in enumerate(fused_tracks):
                if isinstance(track, dict):
                    # Dictionary format from get_fused_tracks()
                    pos = track.get('position', track.get('state', [0, 0])[:2])
                    pos_cov = track.get('covariance', np.eye(4))[:2, :2]
                else:
                    # Assume object with attributes
                    if hasattr(track, 'fused_state') and track.fused_state is not None:
                        pos = track.fused_state[:2]
                        pos_cov = track.fused_cov[:2, :2]
                    elif hasattr(track, 'kf'):
                        state = track.kf.get_state()
                        cov = track.kf.get_covariance()
                        pos = state[:2]
                        pos_cov = cov[:2, :2]
                    else:
                        continue

                # Determine track type and color
                if i in ghost_track_indices:
                    color = 'red'
                    edge_color = 'darkred'
                    marker = 'X'  # X marker for ghost tracks
                    label = 'Fused track (ghost)' if not ghost_track_plotted else ''
                    ghost_track_plotted = True
                    line_style = ':'
                elif i in mis_assoc_track_indices:
                    color = 'orange'
                    edge_color = 'darkorange'
                    marker = 'D'  # Diamond marker for mis-associations
                    label = 'Fused track (mis-associated)' if not mis_assoc_plotted else ''
                    mis_assoc_plotted = True
                    line_style = '--'
                else:
                    color = 'green'
                    edge_color = 'darkgreen'
                    marker = '*'  # Star marker for good tracks
                    label = 'Fused track (correct)' if not good_track_plotted else ''
                    good_track_plotted = True
                    line_style = '-'

                # Plot fused track position
                ax.plot(pos[0], pos[1], marker, color=color, markersize=15,
                        alpha=0.9, markeredgewidth=2, markeredgecolor=edge_color,
                        label=label)

                # Plot 95% confidence ellipse
                width, height, angle = get_confidence_ellipse(pos, pos_cov, confidence=0.95)
                ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle,
                                facecolor='none', edgecolor=color, linewidth=2,
                                linestyle=line_style, alpha=0.7)
                ax.add_patch(ellipse)

                # Label track ID with appropriate color
                ax.text(pos[0] + 1.5, pos[1] - 1.5, f'F{i}',
                        fontsize=8, color=color, fontweight='bold')

            # === Formatting ===
            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)

            time_str = f' (t={frame*DT:.1f}s)'
            ax.set_title(f'Multi-Sensor Tracking - Frame {frame}/{len(self.object_history)-1}{time_str}\n'
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

            # Add legend for uncertainty ellipses and track types
            legend_text = '95% Confidence Ellipses:\n'
            legend_text += 'Dashed = Single sensor\n'
            legend_text += 'Solid/Dotted = Fused tracks\n\n'
            legend_text += 'Fused Track Quality:\n'
            legend_text += 'Star (green) = Correct\n'
            legend_text += 'X (red) = Ghost\n'
            legend_text += 'Diamond (orange) = Mis-assoc'

            ax.text(0.02, 0.98, legend_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

            return []

        # Create animation
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=len(self.object_history),
                           interval=100,  # 100ms per frame
                           blit=False, repeat=True)

        # Save animation
        writer = FFMpegWriter(fps=10, bitrate=1800)  # 10 fps (100ms per frame)
        anim.save(save_path, writer=writer)
        plt.close(fig)

        print(f"Save to: {save_path}")
        return save_path

    def generate_all_animations(self):
        """Generate animations for all vehicles."""
        print(f"\n" + "="*70)
        print(f"Generating Tracking Animations for All Vehicles")
        print("="*70 + "\n")

        saved_files = []
        for v_idx in range(NUM_VEHICLES):
            save_path = f"vehicle_{v_idx}_tracking.mp4"
            self.generate_animation(vehicle_idx=v_idx, save_path=save_path)
            saved_files.append(save_path)

        print(f"\n" + "="*70)
        print(f"All animations created!")
        print("="*70)
        print(f"\nGenerated {len(saved_files)} animation files:")
        for f in saved_files:
            print(f"- {f}")


def main():
    """Main function."""
    np.random.seed(42)  # For reproducibility

    # Run simulation with history
    sim = SimulationWithHistory()
    sim.run()

    # Generate animations for all vehicles
    sim.generate_all_animations()


if __name__ == "__main__":
    main()
