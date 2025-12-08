import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from Object.TwoFingerGripper import TwoFingerGripper
from Object.Objects import Box, Cylinder, Duck, RoboticArm
from Planning.Sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause
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

def addNoiseToOrientation(orientation_quat, roll_noise_range=0.1, pitch_noise_range=0.1, yaw_noise_range=0.1):
    """
    Add uniform random noise to orientation angles (roll, pitch, yaw).
    
    Args:
        orientation_quat: Quaternion orientation [x, y, z, w]
        roll_noise_range: Range for roll noise (in radians). Noise will be uniformly distributed in [-range/2, range/2]
        pitch_noise_range: Range for pitch noise (in radians). Noise will be uniformly distributed in [-range/2, range/2]
        yaw_noise_range: Range for yaw noise (in radians). Noise will be uniformly distributed in [-range/2, range/2]
    
    Returns:
        np.array: Noisy quaternion orientation [x, y, z, w]
    """
    # Convert quaternion to Euler angles
    rotation = R.from_quat(orientation_quat)
    euler_angles = rotation.as_euler("xyz")  # roll, pitch, yaw
    
    # Add uniform random noise in range [-range/2, range/2]
    noisy_euler = euler_angles + np.array([
        np.random.uniform(-roll_noise_range/2, roll_noise_range/2),
        np.random.uniform(-pitch_noise_range/2, pitch_noise_range/2),
        np.random.uniform(-yaw_noise_range/2, yaw_noise_range/2)
    ])
    
    # Convert back to quaternion
    noisy_rotation = R.from_euler("xyz", noisy_euler)
    noisy_quat = noisy_rotation.as_quat(canonical=True)
    
    return noisy_quat

def addNoiseToOffset(grasp_offset, offset_noise_range=0.005):
    """
    Add uniform random noise to grasp offset.
    
    Args:
        grasp_offset: Original grasp offset [x, y, z] in meters
        offset_noise_range: Range for offset noise (in meters). Noise will be uniformly distributed in [-range/2, range/2] for each axis
    
    Returns:
        np.array: Noisy grasp offset [x, y, z]
    """
    noise = np.random.uniform(-offset_noise_range/2, offset_noise_range/2, size=3)
    noisy_offset = np.array(grasp_offset) + noise
    return noisy_offset

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

if __name__ == "__main__":
    gripper_type = "TwoFingerGripper"  # Change to "RoboticArm" to use RoboticArm instead

    if gripper_type == "TwoFingerGripper":
        grasp_data = []

        # Noise ranges for uniform random distribution (noise will be in [-range/2, range/2])
        roll_noise_range = 1
        pitch_noise_range = 1
        yaw_noise_range = 1
        offset_noise_range = 0.06

        plane_id = setupEnvironment()
        gripper_start = np.array([0,0,1])
        object_start = np.array([0,0,0.06])

        s = FibonacciSphere(samples=400, radius=0.6, cone_angle=math.pi)
        s.visualise()
        p.stepSimulation()

        for v in s.vertices:
            # Initialise gripper and object
            gripper = TwoFingerGripper(position=v)
            object = Box(position=object_start)

            if not grasp_data:
                gripper_type = type(gripper).__name__
            
            gripper.load()
            object.load()

            # Let object settle
            for _ in range(50):
                p.stepSimulation()
            
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
                "object_type": "Box",
                "approach_vertex": v.tolist(),
                "grasp_offset": noisy_grasp_offset.tolist()
            }

            grasp_data.append(sample)

            gripper.unload()
            object.unload()

        # Determine object type for filename
        object_type = grasp_data[0]["object_type"] if grasp_data else "Box"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gripper_type}_{object_type}_{timestamp}.csv"
        
        output_file = os.path.join("Samples", filename)
        
        # Convert to DataFrame - flatten nested structure
        records = []
        for sample in grasp_data:
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
            records.append(record)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        
        print(f"\nData collection complete!")
        print(f"Total samples collected: {len(grasp_data)}")
        print(f"Data saved to: {output_file}")
        
        if len(grasp_data) > 0:
            success_count = sum(s["label"] for s in grasp_data)
            print(f"Success rate: {success_count / len(grasp_data) * 100:.2f}%")

        s.removeVisualisation()
 
    elif gripper_type == "RoboticArm":
        
        arm = RoboticArm()
        arm.robotic_arm_grasp_sampling()

    p.disconnect() 