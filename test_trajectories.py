"""Test script to verify trajectory generation."""

import numpy as np
import matplotlib.pyplot as plt
from config import OBJECT_TRAJECTORY_SPECS, OBJECT_INITIAL_STATES
from objects import ObjectSimulator

print("=== Trajectory Generation Verification ===\n")

print("Initial states from OBJECT_INITIAL_STATES:")
for i, state in enumerate(OBJECT_INITIAL_STATES):
    speed = np.sqrt(state[2]**2 + state[3]**2)
    heading_deg = np.rad2deg(np.arctan2(state[3], state[2]))
    print(f"  Object {i+1}: pos=({state[0]:.1f}, {state[1]:.1f}), "
          f"vel=({state[2]:.1f}, {state[3]:.1f}), speed={speed:.2f} m/s, heading={heading_deg:.0f}°")

print("\n" + "="*70 + "\n")

# Initialize simulator with trajectories
sim = ObjectSimulator(trajectory_specs=OBJECT_TRAJECTORY_SPECS)

for i, obj in enumerate(sim.objects):
    print(f"Object {i+1}:")
    spec = OBJECT_TRAJECTORY_SPECS[i]
    print(f"Initial state: {spec['initial_state']}")
    print(f"Maneuvers:")
    for j, maneuver in enumerate(spec['maneuvers']):
        print(f"{j+1}. {maneuver}")

    if obj.trajectory:
        print(f"Generated trajectory length: {len(obj.trajectory)} states")
        print(f"Total duration: {len(obj.trajectory) * 0.1:.1f} seconds")

        # Show first 5 states
        print("  First 5 states:")
        for t, state in enumerate(obj.trajectory[:5]):
            speed = np.sqrt(state[2]**2 + state[3]**2)
            heading_deg = np.rad2deg(np.arctan2(state[3], state[2]))
            print(f"t={t*0.1:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), "
                  f"vel=({state[2]:.2f}, {state[3]:.2f}), speed={speed:.2f} m/s, heading={heading_deg:.0f}°")

        # Show middle states (around maneuver transitions)
        mid = len(obj.trajectory) // 2
        print(f"States around t={mid*0.1:.1f}s:")
        for t in range(max(0, mid-2), min(len(obj.trajectory), mid+3)):
            state = obj.trajectory[t]
            speed = np.sqrt(state[2]**2 + state[3]**2)
            heading_deg = np.rad2deg(np.arctan2(state[3], state[2]))
            print(f"t={t*0.1:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), "
                  f"vel=({state[2]:.2f}, {state[3]:.2f}), speed={speed:.2f} m/s, heading={heading_deg:.0f}°")

        # Show last 5 states
        if len(obj.trajectory) > 5:
            print("Last 5 states:")
            for t, state in enumerate(obj.trajectory[-5:], start=len(obj.trajectory)-5):
                speed = np.sqrt(state[2]**2 + state[3]**2)
                heading_deg = np.rad2deg(np.arctan2(state[3], state[2]))
                print(f"    t={t*0.1:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), "
                      f"vel=({state[2]:.2f}, {state[3]:.2f}), speed={speed:.2f} m/s, heading={heading_deg:.0f}°")

        # Speed verification
        speeds = [np.sqrt(s[2]**2 + s[3]**2) for s in obj.trajectory]
        print(f"Speed stats: min={min(speeds):.2f}, max={max(speeds):.2f}, "
              f"avg={np.mean(speeds):.2f} m/s (constant: {np.std(speeds) < 0.01})")
    else:
        print("No trajectory generated (using constant velocity)")

    print()

# Plot all trajectories
print("\n" + "="*70)
print("Plotting trajectories...")

plt.figure(figsize=(10, 10))
for i, obj in enumerate(sim.objects):
    if obj.trajectory:
        x = [state[0] for state in obj.trajectory]
        y = [state[1] for state in obj.trajectory]
        plt.plot(x, y, label=f'Object {i+1}', linewidth=2)
        plt.scatter(x[0], y[0], s=100, marker='o', zorder=5)  # Start
        plt.scatter(x[-1], y[-1], s=100, marker='s', zorder=5)  # End

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Object Trajectories')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('object_trajectories.png', dpi=150)
print("Saved plot to: object_trajectories.png")
plt.show()
