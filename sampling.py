import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from Grippers.TwoFingerGripper import TwoFingerGripper
from Grippers.RoboticArm import RoboticArm
from Object.Objects import Box, Cylinder, Duck
from Planning.Sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause, addNoiseToOrientation, addNoiseToOffset
from constants import TIME, TICK_RATE, NUM_TICKS

def extractFeatures(gripper, object, approach_vertex, grasp_offset):
    """
    Extract features from the grasp attempt.
    
    Args:
        gripper: Gripper instance
        object: Object instance
        approach_vertex: Approach position (vertex from sphere)
        grasp_offset: Offset from object center (grasp_offset)
    
    Returns:
        dict: Dictionary containing feature values
    """
    object_pos = object.getPosition()
    
    # Get current gripper orientation
    orientation_quat = gripper.getOrientation()
    
    rotation = R.from_quat(orientation_quat)
    euler_angles = rotation.as_euler("xyz")  # roll, pitch, yaw
    
    # Calculate approach direction
    approach_direction = approach_vertex - object_pos
    approach_distance = np.linalg.norm(approach_direction)

    if approach_distance > 0:
        approach_direction = approach_direction / approach_distance
    
    features = {
        "orientation_roll": float(euler_angles[0]),
        "orientation_pitch": float(euler_angles[1]),
        "orientation_yaw": float(euler_angles[2]),
        "offset_x": float(grasp_offset[0]),
        "offset_y": float(grasp_offset[1]),
        "offset_z": float(grasp_offset[2]),
        "approach_dir_x": float(approach_direction[0]),
        "approach_dir_y": float(approach_direction[1]),
        "approach_dir_z": float(approach_direction[2]),
        "approach_distance": float(approach_distance)
    }
    
    return features

def checkGraspSuccess(object, initial_object_pos, threshold=0.15):
    """
    Check if the grasp was successful by checking if object was lifted.
    Note: graspObject already lifts the object, so we check the final position.
    
    Args:
        object: Object instance
        initial_object_pos: Initial object position before grasp
        threshold: Minimum lift distance to consider successful (in meters)
    
    Returns:
        bool: True if grasp was successful, False otherwise
    """
    # Get final object position after graspObject completes
    final_object_pos = np.array(object.getPosition())
    
    # Calculate vertical displacement
    object_lift_distance = final_object_pos[2] - initial_object_pos[2]
    
    # Calculate horizontal displacement (object shouldn't move much horizontally if grasped)
    horizontal_displacement = np.linalg.norm(final_object_pos[:2] - initial_object_pos[:2])
    
    # Success if object lifted above threshold and small horizontal displacement
    lifted_sufficiently = object_lift_distance > threshold
    not_dropped = horizontal_displacement < 0.1
    
    success = lifted_sufficiently and not_dropped
    
    return success

def saveGraspData(grasp_data, gripper_type, object_type="Box"):
    """
    Save grasp data to CSV file in a consistent format for all grippers.
    
    Args:
        grasp_data: List of sample dictionaries
        gripper_type: String name of the gripper type
        object_type: String name of the object type
    """
    if not grasp_data:
        print("No data to save!")
        return
    
    # Determine object type from data if not provided
    if object_type == "Box" and grasp_data:
        object_type = grasp_data[0].get("object_type", "Box")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{gripper_type}_{object_type}_{timestamp}.csv"
    
    # Ensure Samples directory exists
    os.makedirs("Samples", exist_ok=True)
    output_file = os.path.join("Samples", filename)
    
    # Convert to DataFrame - flatten nested structure if needed
    records = []
    for sample in grasp_data:
        # Handle both nested features dict and flat dict formats
        if "features" in sample:
            record = {
                "orientation_roll": sample["features"]["orientation_roll"],
                "orientation_pitch": sample["features"]["orientation_pitch"],
                "orientation_yaw": sample["features"]["orientation_yaw"],
                "offset_x": sample["features"]["offset_x"],
                "offset_y": sample["features"]["offset_y"],
                "offset_z": sample["features"]["offset_z"],
                "approach_dir_x": sample["features"]["approach_dir_x"],
                "approach_dir_y": sample["features"]["approach_dir_y"],
                "approach_dir_z": sample["features"]["approach_dir_z"],
                "approach_distance": sample["features"]["approach_distance"],
                "label": sample["label"],
                "object_type": sample["object_type"]
            }
        else:
            # Already flat format (e.g., from RoboticArm)
            record = sample
        
        records.append(record)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    
    print(f"\nData collection complete!")
    print(f"Total samples collected: {len(grasp_data)}")
    print(f"Data saved to: {output_file}")
    
    if len(grasp_data) > 0:
        success_count = sum(s["label"] if isinstance(s["label"], (int, bool)) else s.get("label", 0) for s in grasp_data)
        print(f"Success rate: {success_count / len(grasp_data) * 100:.2f}%")
    
    return output_file

def main(num_samples:int=500, gripper_type:str="TwoFingerGripper", object_type:str="Box", gui:bool=False):

    if gripper_type == "TwoFingerGripper":
        grasp_data = []

        # Noise ranges for uniform random distribution (noise will be in [-range/2, range/2])
        roll_noise_range = 1
        pitch_noise_range = 1
        yaw_noise_range = 1
        offset_noise_range = 0.06

        plane_id = setupEnvironment(gui=gui)
        gripper_start = np.array([0,0,1])
        object_start = np.array([0,0,0.06])

        s = FibonacciSphere(samples=2*num_samples, radius=0.6, cone_angle=math.pi)
        s.visualise()
        p.stepSimulation()

        print(f"Sampling {len(s.vertices)} vertices using {gripper_type}")

        for idx, v in enumerate(s.vertices):
            if idx % 50 == 0:
                print(f"Sampling {idx + 1} of {len(s.vertices)} vertices")

            # Initialise gripper and object
            gripper = TwoFingerGripper(position=v)

            if object_type == "Box":
                object = Box(position=object_start)
            elif object_type == "Cylinder":
                object = Cylinder(position=object_start)
            elif object_type == "Duck":
                object = Duck(position=object_start)
            else:
                print(f"Unknown object type: {object_type}")
                continue
            
            gripper.load()
            object.load()

            # Let object settle
            for _ in range(50):
                p.stepSimulation()
            
            # Duck needs to be rotated by 90 degrees around the x-axis
            if object_type == "Duck":
                duck_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
                current_position = object.getPosition()
                object.setPosition(new_position=current_position, new_orientation=duck_orientation)
            
            target = object.getPosition()
            orientation = gripper.orientationToTarget(target)
            
            # Add noise to orientation
            noisy_orientation = addNoiseToOrientation(
                orientation, 
                roll_noise_range=roll_noise_range,
                pitch_noise_range=pitch_noise_range,
                yaw_noise_range=yaw_noise_range
            )

            gripper.setPosition(new_position=v, new_orientation=noisy_orientation)
            p.stepSimulation()
            
            gripper.open()
            p.stepSimulation()

            initial_object_pos = object.getPosition()
            
            # Add uniform random noise to grasp offset and extract ml features
            original_grasp_offset = object.grasp_offset
            noisy_grasp_offset = addNoiseToOffset(original_grasp_offset, offset_noise_range=offset_noise_range)
            features = extractFeatures(gripper, object, v, noisy_grasp_offset)
            
            # Pass noisy offset to graspObject so it uses the noisy offset in the actual grasp
            gripper.graspObject(object, grasp_offset=noisy_grasp_offset)
            
            success = checkGraspSuccess(object, initial_object_pos)
            
            sample = {
                "features": features,
                "label": 1 if success else 0,
                "object_type": object_type,
                "approach_vertex": v.tolist(),
                "grasp_offset": noisy_grasp_offset.tolist()
            }

            grasp_data.append(sample)

            gripper.unload()
            object.unload()

        # Save data after all samples collected
        saveGraspData(grasp_data, gripper_type, object_type)
        
        s.removeVisualisation()

    elif gripper_type == "RoboticArm":
        # Use RoboticArm's sampling method but ensure it saves with consistent format
        arm = RoboticArm()
        
        # Call the robotic arm sampling method
        arm.robotic_arm_grasp_sampling(object_type=object_type, gui=gui, num_samples=num_samples)
        
        print("RoboticArm sampling completed. Data saved by RoboticArm class.")

    else:
        print(f"Unknown gripper type: {gripper_type}")
        print("Supported types: 'TwoFingerGripper', 'RoboticArm'")

    p.disconnect()


if __name__ == "__main__":
    main(gripper_type="RoboticArm", object_type="Box", gui=False)