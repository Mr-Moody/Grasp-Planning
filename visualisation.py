import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# CHANGE THIS FILE PATH to your desired CSV
CSV_FILE = os.path.join("Samples", "TwoFingerGripper_Box_20251208_140409.csv")  # <- Edit this line

# Load single CSV file
df = pd.read_csv(CSV_FILE)

# Object center (fixed at origin)
center = np.array([0.0, 0.0, 0.0])

# Compute start positions: center + approach_dir * approach_distance
# (approach_dir points FROM object TO gripper start, so we add)
df['start_x'] = center[0] + df['approach_dir_x'] * df['approach_distance']
df['start_y'] = center[1] + df['approach_dir_y'] * df['approach_distance']
df['start_z'] = center[2] + df['approach_dir_z'] * df['approach_distance']

# Separate success (label=1) and failure (label=0)
success = df[df['label'] == 1]
fail = df[df['label'] == 0]

print(f"Total: {len(df)}, Success: {len(success)}, Fail: {len(fail)}")
print(f"Success rate: {df['label'].mean():.1%}")

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot green arrows for success (thicker, more opaque)
for i in range(len(success)):
    start = [success.iloc[i]['start_x'], success.iloc[i]['start_y'], success.iloc[i]['start_z']]
    end = [center[0], center[1], center[2]]
    ax.quiver(start[0], start[1], start[2], 
              end[0]-start[0], end[1]-start[1], end[2]-start[2],
              color='green', alpha=0.6, linewidth=1.5, arrow_length_ratio=0.1)

# Plot red arrows for failure (thinner, more transparent)
for i in range(len(fail)):
    start = [fail.iloc[i]['start_x'], fail.iloc[i]['start_y'], fail.iloc[i]['start_z']]
    end = [center[0], center[1], center[2]]
    ax.quiver(start[0], start[1], start[2], 
              end[0]-start[0], end[1]-start[1], end[2]-start[2],
              color='red', alpha=0.4, linewidth=0.8, arrow_length_ratio=0.1)

# Mark object center
ax.scatter([0], [0], [0], color='blue', s=100, marker='o', label='Object Center')

# Labels and styling
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Grasp Success/Failure: {CSV_FILE}\nGreen=Success, Red=Failure')

# Equal aspect ratio bounds
max_range = 0.7
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Legend and grid
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
# Extract filename without extension for the output image
csv_basename = os.path.splitext(os.path.basename(CSV_FILE))[0]
plots_dir = "Plots"
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'grasp_arrows_3d_{csv_basename}.png'), dpi=300, bbox_inches='tight')
plt.show()
