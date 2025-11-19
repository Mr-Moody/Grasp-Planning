"""
Use trained model to predict successful grasps.
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from train_grasp_model import loadModel, predictGrasp
from Object.Gripper import Gripper
from util import setupEnvironment

def predictGraspFromGripperObject(gripper: Gripper, object_pos: np.ndarray, 
                                      approach_vertex: np.ndarray, grasp_offset: np.ndarray):
    """
    Predict if a grasp will be successful given gripper and object information.
    
    Args:
        gripper: Gripper instance
        object_pos: Object position (numpy array)
        approach_vertex: Approach position (numpy array)
        grasp_offset: Offset from object center (numpy array)
    
    Returns:
        tuple: (prediction, probability) where prediction is 0 (failure) or 1 (success)
    """
    # Load trained model
    try:
        model, scaler = loadModel()
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first using train_grasp_model.py")
        return None, None

    gripper.load()
    
    # Calculate approach orientation
    orientation_quat = gripper.orientationToTarget(object_pos)
    rotation = R.from_quat(orientation_quat)
    euler_angles = rotation.as_euler('xyz')
    
    # Calculate approach direction
    approach_direction = approach_vertex - object_pos
    approach_distance = np.linalg.norm(approach_direction)
    if approach_distance > 0:
        approach_direction = approach_direction / approach_distance
    
    # Make prediction
    prediction, probability = predictGrasp(
        model, scaler,
        orientation_roll=euler_angles[0],
        orientation_pitch=euler_angles[1],
        orientation_yaw=euler_angles[2],
        offset_x=grasp_offset[0],
        offset_y=grasp_offset[1],
        offset_z=grasp_offset[2],
        approach_dir_x=approach_direction[0],
        approach_dir_y=approach_direction[1],
        approach_dir_z=approach_direction[2],
        approach_distance=approach_distance
    )

    gripper.unload()
    
    return prediction, probability

def findBestGrasp(gripper: Gripper, object_pos: np.ndarray, 
                    approach_vertices: list, grasp_offsets: list):

    best_grasp = None
    best_probability = -1
    
    for vertex in approach_vertices:
        for offset in grasp_offsets:
            prediction, probability = predictGraspFromGripperObject(
                gripper, object_pos, vertex, offset
            )
            
            if probability is not None and probability[1] > best_probability:
                best_probability = probability[1]
                best_grasp = {
                    "approach_vertex": vertex,
                    "grasp_offset": offset,
                    "prediction": prediction,
                    "probability": probability,
                    "success_probability": probability[1]
                }
    
    return best_grasp

def main():
    from Object.TwoFingerGripper import TwoFingerGripper
    
    gripper = TwoFingerGripper()
    object_pos = np.array([1, 0, 0.06])
    approach_vertex = np.array([0, 0, 0.6])
    grasp_offset = np.array([0, 0, 0.01])
    
    prediction, probability = predictGraspFromGripperObject(
        gripper, object_pos, approach_vertex, grasp_offset
    )

    print("Prediction: ", prediction)
    print("Probability: ", probability)
    
    if prediction is not None:
        print(f"Prediction: {'Success' if prediction == 1 else 'Failure'}")
        print(f"Success Probability: {probability[1]:.4f}")
        print(f"Failure Probability: {probability[0]:.4f}")
    else:
        print("Could not make prediction. Please train the model first.")

if __name__ == "__main__":
    setupEnvironment()
    main()

