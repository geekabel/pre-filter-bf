### File Structure

- **`config.py`**: All simulation parameters (sensor noise, DS thresholds, association gates)
- **`main.py`**: Simulation orchestration and logging
- **`objects.py`**: Ground truth object simulator (constant velocity model)
- **`sensors.py`**: Sensor models with AR-1 time-correlated noise
  - Camera: high noise (5, 6, 2, 2.4), AR coeff=0.7
  - Radar: medium noise (3, 2.5, 1.2, 1.5), AR coeff=0.5
  - Lidar: low noise (1, 0.7, 0.1, 0.07), AR coeff=0.3
- **`vehicle_agent.py`**: Per-vehicle agent coordinating sensors and fusion
- **`single_sensor_tracker.py`**: Single-sensor tracker implementing Lima's sensor-level loop
- **`ds_lima.py`**: Lima's DS existence reasoning (4 operations: birth, discount, update, delete)
- **`kalman_filter.py`**: Constant-velocity Kalman filter + Hungarian M2T association
- **`greedy_multidim.py`**: Greedy pairwise GNN for T2T clustering
- **`multi_sensor_fusion.py`**: Track clustering + Covariance Intersection fusion
- **`evaluation.py`**: Metrics for ghost tracks, false associations, tracking error
- **`visualization.py`**: Plot utilities
- **`logger.py`**: CSV logging for metrics

### Run

```python
python main.py
```
