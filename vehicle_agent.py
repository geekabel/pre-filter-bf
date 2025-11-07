"""Vehicle agent with multi-sensor tracking capability.

Each vehicle has:
- Multiple sensors (camera, radar, lidar)
- Single-sensor trackers (one per sensor)
- Multi-sensor fusion module
"""

from sensors import SensorArray
from single_sensor_tracker import SingleSensorTracker
from multi_sensor_fusion import MultiSensorFusion
from association_logger import AssociationLogger
from config import SENSOR_TYPES


class VehicleAgent:
    """Vehicle agent with sensor array and tracking system."""

    def __init__(self, vehicle_id, enable_association_logging=True):
        """Initialize vehicle agent.

        Args:
            vehicle_id: Unique vehicle identifier
            enable_association_logging: If True, log Hungarian associations
        """
        self.vehicle_id = vehicle_id

        # Sensor array
        self.sensor_array = SensorArray(vehicle_id)

        # Association logger (shared by all sensors on this vehicle)
        self.association_logger = None
        if enable_association_logging:
            self.association_logger = AssociationLogger(
                output_path=f"out/association_log_vehicle_{vehicle_id}.csv"
            )

        # Single-sensor trackers (one per sensor)
        self.single_sensor_trackers = [
            SingleSensorTracker(sensor_id, SENSOR_TYPES[sensor_id], logger=self.association_logger)
            for sensor_id in range(len(SENSOR_TYPES))
        ]

        # Multi-sensor fusion
        self.fusion = MultiSensorFusion(vehicle_id)

    def process_timestep(self, object_states, timestep=None):
        """Process one time step.

        Pipeline:
        a) Generate measurements for each object from each sensor
        b) Feed measurements to corresponding single-sensor tracker
        c) Each single-sensor tracker runs Lima's sensor-level loop
        d) Fuse tracks from all sensors

        Args:
            object_states: List of true object states [x, y, vx, vy]
            timestep: Optional timestep for logging

        Returns:
            Fused tracks from multi-sensor fusion
        """
        # Step a) Generate measurements from all sensors
        measurements_by_sensor = self.sensor_array.measure_objects(object_states)

        # Step b-c) Process measurements through single-sensor trackers
        for sensor_id, tracker in enumerate(self.single_sensor_trackers):
            # Get detections for this sensor
            detections = measurements_by_sensor[sensor_id]
            # Process through single-sensor tracker
            tracker.process_detections(detections, timestep=timestep, true_states=object_states)

        # Step d) Multi-sensor fusion
        sensor_tracks = [tracker.get_confirmed_tracks()
                        for tracker in self.single_sensor_trackers] # we keep only 'confirmed' tracks (filtered by DS existence module)
        fused_tracks = self.fusion.fuse_sensor_tracks(sensor_tracks)

        return fused_tracks

    def get_single_sensor_tracks(self, sensor_id):
        """Get tracks from a specific sensor.

        Args:
            sensor_id: Sensor identifier (0=camera, 1=radar, 2=lidar)

        Returns:
            List of track states for this sensor
        """
        return self.single_sensor_trackers[sensor_id].get_track_states()

    def get_all_single_sensor_tracks(self):
        """Get tracks from all sensors.

        Returns:
            Dict mapping sensor_id -> list of track states
        """
        return {
            sensor_id: tracker.get_track_states()
            for sensor_id, tracker in enumerate(self.single_sensor_trackers)
        }

    def get_fused_tracks(self):
        """Get fused tracks."""
        return self.fusion.get_fused_tracks()

    def get_num_tracks_per_sensor(self):
        """Get number of tracks maintained by each sensor.

        Returns:
            Dict mapping sensor_type -> num_tracks
        """
        return {
            SENSOR_TYPES[sensor_id]: tracker.get_num_tracks()
            for sensor_id, tracker in enumerate(self.single_sensor_trackers)
        }

    def get_num_fused_tracks(self):
        """Get number of fused tracks."""
        return self.fusion.get_num_fused_tracks()

    def get_statistics(self):
        """Get tracking statistics.

        Returns:
            Dict with various statistics
        """
        num_tracks_per_sensor = self.get_num_tracks_per_sensor()
        num_fused = self.get_num_fused_tracks()

        # Count confirmed tracks per sensor
        confirmed_per_sensor = {}
        for sensor_id, tracker in enumerate(self.single_sensor_trackers):
            confirmed = len(tracker.get_confirmed_tracks())
            confirmed_per_sensor[SENSOR_TYPES[sensor_id]] = confirmed

        return {
            'vehicle_id': self.vehicle_id,
            'num_tracks_per_sensor': num_tracks_per_sensor,
            'num_fused_tracks': num_fused,
            'confirmed_per_sensor': confirmed_per_sensor
        }

    def reset(self):
        """Reset all sensors and trackers."""
        self.sensor_array.reset()
        for tracker in self.single_sensor_trackers:
            tracker.tracks = []
            tracker.next_track_id = 0
        self.fusion.fused_tracks = []
