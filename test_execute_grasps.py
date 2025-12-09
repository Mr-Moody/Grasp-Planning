import pybullet as p
import numpy as np
import math
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Grippers.TwoFingerGripper import TwoFingerGripper
from Grippers.RoboticArm import RoboticArm
from Object.Objects import Box, Cylinder, Duck
from Planning.Sphere import FibonacciSphere
from util import setupEnvironment
from sampling import extractFeatures, checkGraspSuccess
from predict_grasp import predictGraspFromGripperObject, _calculateOrientationToTarget
from train_grasp_model import loadModel, predictGrasp
from constants import TICK_RATE

def generateTestGrasps(num_grasps:int=10, gripper_type:str="TwoFingerGripper", object_type:str="Box", seed:int=42) -> List[Dict]:
    """
    Generate new test grasp configurations that are not in the training set.
    
    Args:
        num_grasps: Number of test grasps to generate
        gripper_type: Type of gripper to use
        object_type: Type of object to grasp
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries containing approach_vertex and grasp_offset for each test grasp
    """
    np.random.seed(seed)
    
    sphere = FibonacciSphere(samples=400, radius=0.6, cone_angle=math.pi)
    
    # Randomly select num_grasps vertices from the sphere
    selected_indices = np.random.choice(len(sphere.vertices), size=min(num_grasps, len(sphere.vertices)), replace=False)
    
    test_grasps = []
    for idx in selected_indices:
        approach_vertex = sphere.vertices[idx]
        
        test_grasps.append({
            "approach_vertex": approach_vertex,
            "grasp_offset": np.array([0, 0, 0.01])
        })
    
    return test_grasps

def executeTestGrasp(gripper, object, approach_vertex:np.ndarray, grasp_offset:np.ndarray, 
                    object_type:str, gui:bool=False) -> Tuple[bool, Dict]:
    """
    Execute a single test grasp in simulation and determine if it was successful.
    
    Args:
        gripper: Gripper instance
        object: Object instance
        approach_vertex: Approach position
        grasp_offset: Offset from object center
        object_type: Type of object
        gui: Whether to show GUI
    
    Returns:
        Tuple of (success: bool, features: dict)
    """
    origin_position = np.array([0, 0, 0.06])
    
    # Set orientation based on object type
    if object_type == "Duck":
        origin_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
    else:
        origin_orientation = np.array([0, 0, 0, 1])
    
    # Reset object position and orientation
    object.setPosition(new_position=origin_position, new_orientation=origin_orientation)
    
    # Reset object velocity to zero to ensure it's at rest
    p.resetBaseVelocity(object.id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    
    # Let object settle at origin
    for _ in range(100):
        p.stepSimulation()
    
    object_pos = object.getPosition()
    
    # Set gripper position and orientation
    gripper_orientation = gripper.orientationToTarget(object_pos)
    gripper.setPosition(new_position=approach_vertex, new_orientation=gripper_orientation)
    
    # Let gripper settle
    for _ in range(20):
        p.stepSimulation()
    
    gripper.open()
    p.stepSimulation()
    
    initial_object_pos = object.getPosition()
    
    features = extractFeatures(gripper, object, approach_vertex, grasp_offset)
    
    gripper.graspObject(object, grasp_offset=grasp_offset)
    
    success = checkGraspSuccess(object, initial_object_pos)
    
    return success, features


def testClassifierPredictions(num_grasps: int = 10, gripper_type: str = "TwoFingerGripper",
                              object_type: str = "Box", gui: bool = False, seed: int = 42) -> Dict:
    """
    Test classifier predictions by executing grasps and comparing predictions to actual results.
    
    Args:
        num_grasps: Number of test grasps to execute
        gripper_type: Type of gripper to use
        object_type: Type of object to grasp
        gui: Whether to show GUI during execution
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing test results and metrics
    """
    print("=" * 60)
    print("Classifier Test Execution")
    print("=" * 60)
    print(f"Gripper Type: {gripper_type}")
    print(f"Object Type: {object_type}")
    print(f"Number of Test Grasps: {num_grasps}\n")

    # Load trained model
    try:
        model, scaler = loadModel()
        print("Loaded trained model")
    except FileNotFoundError:
        print("ERROR: Model not found. Please train the model first using train_classifier mode.")
        return None
    
    from scipy.spatial.transform import Rotation as R
    
    setupEnvironment(gui=gui)
    
    # Generate test grasps
    print(f"Generating {num_grasps} test grasps...")
    test_grasps = generateTestGrasps(num_grasps, gripper_type, object_type, seed)
    print(f"Generated {len(test_grasps)} test grasps")
    print()
    
    # Initialise gripper and object
    if gripper_type == "TwoFingerGripper":
        gripper = TwoFingerGripper()
    elif gripper_type == "RoboticArm":
        gripper = RoboticArm()
    else:
        print(f"ERROR: Unknown gripper type: {gripper_type}")
        return None
    
    if object_type == "Box":
        object = Box(position=np.array([0, 0, 0.06]))
    elif object_type == "Cylinder":
        object = Cylinder(position=np.array([0, 0, 0.06]))
    elif object_type == "Duck":
        object = Duck(position=np.array([0, 0, 0.06]))
    else:
        print(f"ERROR: Unknown object type: {object_type}")
        return None
    
    gripper.load()
    object.load()
    
    predictions = []
    actual_results = []
    prediction_probs = []
    test_results = []
    
    print("Executing test grasps...")
    for i, test_grasp in enumerate(test_grasps):
        print(f"  Test grasp {i+1}/{len(test_grasps)}...", end=" ")
        
        approach_vertex = test_grasp["approach_vertex"]
        grasp_offset = test_grasp["grasp_offset"]
        object_pos = object.getPosition()
        
        orientation_quat = _calculateOrientationToTarget(approach_vertex, object_pos)
        rotation = R.from_quat(orientation_quat)
        euler_angles = rotation.as_euler('xyz')
        
        # Calculate approach direction
        approach_direction = approach_vertex - object_pos
        approach_distance = np.linalg.norm(approach_direction)
        if approach_distance > 0:
            approach_direction = approach_direction / approach_distance
        
        # Make prediction using model
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
        
        # Execute grasp and get actual result
        success, features = executeTestGrasp(gripper, object, approach_vertex, grasp_offset, object_type, gui)
        
        predictions.append(prediction)
        actual_results.append(1 if success else 0)
        prediction_probs.append(probability[1] if probability is not None else 0.0)
        
        test_results.append({
            "grasp_id": i + 1,
            "prediction": prediction,
            "actual": 1 if success else 0,
            "success_probability": probability[1] if probability is not None else 0.0,
            "correct": prediction == (1 if success else 0),
            "features": features
        })
        
        gripper.unload()
        object.unload()
        
        gripper.load()
        if object_type == "Box":
            object = Box(position=np.array([0, 0, 0.06]))
        elif object_type == "Cylinder":
            object = Cylinder(position=np.array([0, 0, 0.06]))
        elif object_type == "Duck":
            object = Duck(position=np.array([0, 0, 0.06]))
        object.load()
        
        # Let object settle
        for _ in range(50):
            p.stepSimulation()
        
        print(f"Predicted: {'Success' if prediction == 1 else 'Failure'}\nActual: {'Success' if success else 'Failure'}")
    
    gripper.unload()
    object.unload()
    p.disconnect()
    
    # Calculate metrics
    predictions_array = np.array(predictions)
    actual_array = np.array(actual_results)
    
    accuracy = accuracy_score(actual_array, predictions_array)
    cm = confusion_matrix(actual_array, predictions_array, labels=[0, 1])
    
    # Calculate additional metrics
    correct_predictions = np.sum(predictions_array == actual_array)
    total_predictions = len(predictions_array)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((predictions_array == 1) & (actual_array == 1))
    fp = np.sum((predictions_array == 1) & (actual_array == 0))
    tn = np.sum((predictions_array == 0) & (actual_array == 0))
    fn = np.sum((predictions_array == 0) & (actual_array == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Total Test Grasps: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print("Confusion Matrix:")
    print("                 Predicted")
    print("              Failure  Success")
    print(f"Actual Failure   {tn:4d}     {fp:4d}")
    print(f"       Success   {fn:4d}     {tp:4d}\n")
    print("Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1_score:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(actual_array, predictions_array, target_names=["Failure", "Success"],zero_division=0.0))
    
    # Detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for result in test_results:

        print(f"Grasp {result['grasp_id']:2d}: | "
              f"Predicted: {'Success' if result['prediction'] == 1 else 'Failure':7s} | "
              f"Actual: {'Success' if result['actual'] == 1 else 'Failure':7s} | "
              f"Prob: {result['success_probability']:.3f}")
    
    print("=" * 60)
    
    return {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "test_results": test_results,
        "predictions": predictions,
        "actual_results": actual_results
    }


if __name__ == "__main__":
    results = testClassifierPredictions(
        num_grasps=10,
        gripper_type="TwoFingerGripper",
        object_type="Box",
        gui=False,
        seed=42
    )

