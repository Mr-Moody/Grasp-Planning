import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import cast
import os
import seaborn as sns

def visualiseGrasps(csv_file: str):
    df = pd.read_csv(csv_file)

    center = np.array([0.0, 0.0, 0.0])

    df["start_x"] = center[0] + df["approach_dir_x"] * df["approach_distance"]
    df["start_y"] = center[1] + df["approach_dir_y"] * df["approach_distance"]
    df["start_z"] = center[2] + df["approach_dir_z"] * df["approach_distance"]

    # Separate success (label=1) and failure (label=0)
    success = df[df["label"] == 1]
    fail = df[df["label"] == 0]

    print(f"Total: {len(df)}, Success: {len(success)}, Fail: {len(fail)}")
    success_rate = df["label"].mean()
    print(f"Success rate: {success_rate:.1%}")

    arrow_length = 0.075

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

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
                color="green", alpha=0.6, linewidth=1.5, arrow_length_ratio=0.1)

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
    ax.scatter(0, 0, 0, color="blue", s=100, marker="o", label="Object Center")

    # Labels and styling
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Grasp Success/Failure: {os.path.basename(csv_file)}\nGreen=Success, Red=Failure")

    # Legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()
    
def visualiseConfusionMatrix(confusion_matrix:np.ndarray, save_dir:str="Plots"):
    """
    Visualise and save the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=sns.cubehelix_palette(as_cmap=True), 
                xticklabels=["Failure", "Success"], 
                yticklabels=["Failure", "Success"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix") #
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    #visualiseGrasps("Samples/TwoFingerGripper_Duck_20251208_231009.csv")
    visualiseConfusionMatrix(np.array([[1, 0], [2, 7]]), "Plots")