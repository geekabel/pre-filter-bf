"""Main simulation script for multi-vehicle tracking with Lima's DS sensor fusion."""

import numpy as np
from config import (NUM_VEHICLES, NUM_TIME_STEPS,
                   OBJECT_TRAJECTORY_SPECS, NUM_OBJECTS, SENSOR_TYPES, DT)
from objects import ObjectSimulator
from vehicle_agent import VehicleAgent
from evaluation import TrackingEvaluator
from visualization import plot_track_counts_vs_truth
from logger import SimulationLogger


def run_simulation():
    """Run the complete multi-vehicle tracking simulation."""
    print("\n" + "="*70)
    print("Multi-Vehicle Tracking with Lima's DS Sensor-Level Existence")
    print("="*70 + "\n")

    # Initialize object simulator
    object_sim = ObjectSimulator(trajectory_specs=OBJECT_TRAJECTORY_SPECS)
    print(f"Initialized {NUM_OBJECTS} objects with a constant velocity")

    # Initialize vehicle agents
    vehicles = [VehicleAgent(i) for i in range(NUM_VEHICLES)]
    print(f"Initialized {NUM_VEHICLES} vehicles with {len(SENSOR_TYPES)} sensors each")
    print(f"Sensors: {', '.join(SENSOR_TYPES)}")

    # Initialize evaluator and logger
    evaluator = TrackingEvaluator(NUM_OBJECTS)
    logger = SimulationLogger(output_dir="out")

    sensor_confirmed_history = {sensor: [] for sensor in SENSOR_TYPES}
    fused_count_history = []
    time_steps = []

    print(f"\nRunning simulation for {NUM_TIME_STEPS} time steps (DT=100ms)...")
    print("-" * 70)

    # Main simulation loop
    for t in range(NUM_TIME_STEPS):
        time_sec = t * DT  # Convert timestep to seconds

        # Get true object states
        true_states = object_sim.get_states()

        # Process each vehicle
        for vehicle in vehicles:
            # Run pipeline: sensors → single-sensor trackers → fusion
            fused_tracks = vehicle.process_timestep(true_states, timestep=t)

        vehicle_0 = vehicles[0]
        stats_vehicle0 = vehicle_0.get_statistics()
        time_steps.append(t)
        fused_count_history.append(stats_vehicle0['num_fused_tracks'])
        for sensor_type in SENSOR_TYPES:
            confirmed = stats_vehicle0['confirmed_per_sensor'].get(sensor_type, 0)
            sensor_confirmed_history[sensor_type].append(confirmed)

        # Evaluate tracking performance (using fused tracks)
        evaluator.evaluate_timestep(t, true_states, vehicles)

        # Log metrics (fleet-level aggregate)
        history = evaluator.get_history()
        if history['overall_metrics']:
            latest = history['overall_metrics'][-1]
            metrics = {
                'num_tracks': latest['num_tracks'],
                'ghost_tracks': latest['ghost_tracks'],
                'missed_objects': latest['missed_objects'],
                'tracking_error': latest['tracking_error'],
                'false_associations': latest['false_associations'],
                'rmse_tracking_error': latest['rmse_tracking_error'],
                'vehicles_with_four_tracks': latest['vehicles_with_four_tracks'],
                'perfect_vehicles': latest['perfect_vehicles']
            }
            logger.log_metrics(t, time_sec, metrics)

        # Log sensor existence summary per vehicle/sensor
        for vehicle in vehicles:
            stats = vehicle.get_statistics()
            vehicle_id = stats['vehicle_id']
            fused_count = stats['num_fused_tracks']
            for sensor_type, total_tracks in stats['num_tracks_per_sensor'].items():
                confirmed_tracks = stats['confirmed_per_sensor'].get(sensor_type, 0)
                logger.log_sensor_summary(
                    t, time_sec, vehicle_id, sensor_type,
                    total_tracks, confirmed_tracks, fused_count
                )

        # Print progress every 10 steps
        if (t + 1) % 10 == 0:
            stats = vehicles[0].get_statistics()
            print(f"Step {t+1:3d}: Vehicle 0 - ", end="")
            print(f"Sensors: {stats['num_tracks_per_sensor']}, ", end="")
            print(f"Fused: {stats['num_fused_tracks']} tracks")

        # Update object positions
        object_sim.step()

    print("-" * 70)
    print("Simulation complete!\n")

    # Print evaluation summary
    evaluator.print_summary()

    # Print detailed sensor statistics
    print("\n=== Sensor-Level Statistics (Vehicle 0) ===")
    for sensor_id, sensor_type in enumerate(SENSOR_TYPES):
        tracker = vehicles[0].single_sensor_trackers[sensor_id]
        tracks = tracker.get_track_states()
        confirmed = [t for t in tracks if t['status'] == 'confirmed']
        tentative = [t for t in tracks if t['status'] == 'tentative']
        decaying = [t for t in tracks if t['status'] == 'decaying']

        print(f"\n{sensor_type.capitalize()} sensor:")
        print(f"Total tracks: {len(tracks)}")
        print(f"Confirmed: {len(confirmed)}")
        print(f"Tentative: {len(tentative)}")
        print(f"Decaying: {len(decaying)}")

        if len(tracks) > 0:
            avg_belief = np.mean([t['existence']['belief'] for t in tracks])
            avg_pignistic = np.mean([t['existence']['pignistic'] for t in tracks])
            print(f"Avg Belief(E): {avg_belief:.3f}")
            print(f"Avg Pignistic(E): {avg_pignistic:.3f}")

    print("\n" + "="*40)

    # Generate visual diagnostic: track counts vs truth
    print("\nGenerating track count diagnostics for Vehicle 0...")
    plot_track_counts_vs_truth(
        time_steps,
        sensor_confirmed_history,
        fused_count_history,
        NUM_OBJECTS,
        save_path='track_counts_vs_truth.png'
    )

    if time_steps:
        print("\nFinal confirmed track counts (Vehicle 0):")
        print(f"Fused tracks: {fused_count_history[-1]}")
        for sensor_type in SENSOR_TYPES:
            print(f"{sensor_type.capitalize()} sensor: {sensor_confirmed_history[sensor_type][-1]}")

    # Save CSV logs
    logger.save_all()

    return evaluator, (time_steps, sensor_confirmed_history, fused_count_history), vehicles


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    evaluator, diagnostics, vehicles = run_simulation()
