"""Configuration parameters for the multi-vehicle tracking simulation."""

import numpy as np

# Simulation parameters
NUM_VEHICLES = 3
NUM_SENSORS_PER_VEHICLE = 3
NUM_OBJECTS = 4
NUM_TIME_STEPS = 50
DT = 0.1  # Time step in seconds (100ms updates)

# Sensor types
SENSOR_TYPES = ['camera', 'radar', 'lidar']

# Sensor noise parameters using VNC model
# Format: [sigma_x, sigma_y, sigma_vx, sigma_vy] in meters and m/s
# From VNC paper: low=(1, 0.7, 0.1, 0.07), high=(5, 6, 2, 2.4), medium=(3, 2.5, 1.2, 1.5)
SENSOR_NOISE_VNC = {
    'lidar': np.array([1.0, 0.7, 0.1, 0.07]),      # Low noise
    'radar': np.array([3.0, 2.5, 1.2, 1.5]),       # Medium noise
    'camera': np.array([5.0, 6.0, 2.0, 2.4])       # High noise
}

# AR-1 Markov process parameters for time-correlated noise
# Higher coefficient = more correlation between consecutive time steps
CAMERA_AR_COEFFICIENT = 0.7   # Camera: high correlation (noisy, drifts slowly)
RADAR_AR_COEFFICIENT = 0.5    # Radar: moderate correlation
LIDAR_AR_COEFFICIENT = 0.3    # Lidar: low correlation (most independent)

# Mahalanobis distance threshold for association (SQUARED Mahalanobis distance)
# Chi-squared critical values for 4 DOF (x, y, vx, vy):
MAH_DISTANCE_THRESHOLD = 9.488  # χ²(4 DOF, 95% confidence)

# Lima DS sensor-level existence parameters (Table 3.4)
E_BIRTH = 0.4          # Birth evidence of existence
E_FORGET = 0.6         # Ignorance threshold for deletion
T_EXISTS_HALF = 0.5    # Half-life of existence evidence in seconds (halved for faster decay)
E_UPDATE = 0.7         # Existence increase on association (keep same)

# Confirmation thresholds for track lifecycle
TAU_CONF = 0.75        # Belief threshold to confirm track
TAU_HOLD = 0.3         # Belief threshold to keep tentative track

# Kalman Filter process noise (for velocity components)
PROCESS_NOISE_STD = 0.1  # Process noise standard deviation

# Vehicle positions (fixed for this simulation)
VEHICLE_POSITIONS = np.array([
    [0, 0],
    [10, 0],
    [0, 10]
])

# Initial object positions and velocities
OBJECT_INITIAL_STATES = np.array([
    [5, 5, 0.5, 0.5],    # [x, y, vx, vy] - object 1
    [15, 5, -0.3, 0.2],   # object 2
    [5, 15, 0.2, -0.3],   # object 3
    [15, 15, -0.4, -0.4]  # object 4
])
