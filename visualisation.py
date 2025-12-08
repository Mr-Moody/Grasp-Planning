import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

CSV_FILE = os.path.join("Samples", "TwoFingerGripper_Box_20251208_150831.csv")  # <- Edit this line

# Load single CSV file
df = pd.read_csv(CSV_FILE)

# Object center (fixed at origin)
center = np.array([0.0, 0.0, 0.0])

# Compute start positions: center + approach_dir * approach_distance
# (approach_dir points FROM object TO gripper start, so we add)
df["start_x"] = center[0] + df["approach_dir_x"] * df["approach_distance"]
df["start_y"] = center[1] + df["approach_dir_y"] * df["approach_distance"]
df["start_z"] = center[2] + df["approach_dir_z"] * df["approach_distance"]

# Separate success (label=1) and failure (label=0)
success = df[df["label"] == 1]
fail = df[df["label"] == 0]

print(f"Total: {len(df)}, Success: {len(success)}, Fail: {len(fail)}")
print(f"Success rate: {df['label'].mean():.1%}")

arrow_length = 0.075

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot green arrows for success (thicker, more opaque)
for i in range(len(success)):
    start = np.array([success.iloc[i]["start_x"], success.iloc[i]["start_y"], success.iloc[i]["start_z"]])

    # Direction from vertex to center
    direction = center - start

    # Normalize and scale to arrow_length
    direction_norm = np.linalg.norm(direction)

    if direction_norm > 0:
        direction = direction / direction_norm * arrow_length

    ax.quiver(start[0], start[1], start[2], 
              direction[0], direction[1], direction[2],
              color='green', alpha=0.6, linewidth=1.5, arrow_length_ratio=0.1)

# Plot red arrows for failure (thinner, more transparent)
for i in range(len(fail)):
    start = np.array([fail.iloc[i]["start_x"], fail.iloc[i]["start_y"], fail.iloc[i]["start_z"]])

    # Direction from vertex to center
    direction = center - start

    # Normalize and scale to arrow_length
    direction_norm = np.linalg.norm(direction)

    if direction_norm > 0:
        direction = direction / direction_norm * arrow_length

    ax.quiver(start[0], start[1], start[2], 
              direction[0], direction[1], direction[2],
              color="red", alpha=0.4, linewidth=0.8, arrow_length_ratio=0.1)

# Mark object center
ax.scatter([0], [0], [0], color="blue", s=100, marker="o", label="Object Center")

# Labels and styling
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"Grasp Success/Failure: {CSV_FILE}\nGreen=Success, Red=Failure")

# Legend and grid
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis("equal")
plt.tight_layout()
plt.show()
