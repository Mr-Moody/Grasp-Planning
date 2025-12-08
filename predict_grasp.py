"""
Use trained model to predict successful grasps.
"""
import numpy as np
import pybullet as p
import os
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
from train_grasp_model import loadModel, predictGrasp
from Grippers.Gripper import Gripper
from util import setupEnvironment

# Process-local model cache for multiprocessing
_model_cache = {}
_scaler_cache = {}

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

def _calculateOrientationToTarget(from_pos, target_pos):
    """
    Calculate quaternion orientation to face target from a given position.
    This is a standalone function that doesn't require a gripper instance.
    
    Args:
        from_pos: Starting position (numpy array)
        target_pos: Target position (numpy array)
    
    Returns:
        np.array: Quaternion orientation [x, y, z, w]
    """
    direction = target_pos - from_pos
    
    norm_x = np.linalg.norm(direction)
    
    if norm_x < 1e-6:
        return R.identity().as_quat(canonical=True)
    
    new_x = direction / norm_x
    
    world_up = np.array([0, 0, 1])
    
    new_y = np.cross(world_up, new_x)
    
    if np.linalg.norm(new_y) < 1e-6:
        # if looking up/down manually set new y to the world Y
        new_y = np.cross(new_x, [0, 1, 0])
    
    new_y = new_y / np.linalg.norm(new_y)
    
    new_z = np.cross(new_x, new_y)
    
    rotation_matrix = np.column_stack((new_x, new_y, new_z))
    
    rotation = R.from_matrix(rotation_matrix)
    
    quaternion_xyzw = rotation.as_quat(canonical=True)
    
    return quaternion_xyzw

def _get_model_and_scaler():
    """
    Get model and scaler with process-local caching for multiprocessing.
    Each worker process loads the model once and caches it.
    Sets n_jobs=1 to avoid sklearn multiprocessing warnings.
    """
    process_id = os.getpid()
    
    if process_id not in _model_cache:
        try:
            model, scaler = loadModel()
            # Set n_jobs=1 to avoid sklearn multiprocessing warnings when in multiprocessing context
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=1)
            _model_cache[process_id] = model
            _scaler_cache[process_id] = scaler
        except FileNotFoundError:
            # Return None if model not found - will be handled in objective function
            return None, None
    
    return _model_cache[process_id], _scaler_cache[process_id]

def _objective_function(x, object_pos):
    """
    Objective function for optimization (minimize negative success probability).
    This is a module-level function so it can be pickled for multiprocessing.
    Model is loaded per-process using a cache to avoid pickling issues.
    
    Args:
        x: 6D vector [approach_vertex_x, approach_vertex_y, approach_vertex_z,
                     grasp_offset_x, grasp_offset_y, grasp_offset_z]
        object_pos: Object position
    
    Returns:
        float: Negative success probability (to minimize)
    """
    # Load model (cached per process)
    model, scaler = _get_model_and_scaler()
    
    if model is None or scaler is None:
        # Return high value if model not available
        return 1.0
    
    approach_vertex = np.array(x[:3])
    grasp_offset = np.array(x[3:6])
    
    # Calculate orientation directly from approach_vertex to object_pos
    orientation_quat = _calculateOrientationToTarget(approach_vertex, object_pos)
    rotation = R.from_quat(orientation_quat)
    euler_angles = rotation.as_euler('xyz')
    
    # Calculate approach direction
    approach_direction = approach_vertex - object_pos
    approach_distance = np.linalg.norm(approach_direction)
    if approach_distance > 0:
        approach_direction = approach_direction / approach_distance
    
    # Make prediction
    try:
        _, probability = predictGrasp(
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
        
        # Return negative success probability (maximising success probability and optimizer minimises)
        return -probability[1]
    except Exception as e:
        # Return high value if prediction fails
        return 1.0

def findBestGrasp(gripper: Gripper, object_pos: np.ndarray, 
                    approach_vertex_bounds: tuple = ((-1.0, 1.0), (-1.0, 1.0), (0.3, 1.0)),
                    grasp_offset_bounds: tuple = ((-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.1)),
                    maxiter: int = 100, seed: int = 42, tol: float = 0.01):
    """
    Find the best grasp using optimisation.
    
    Args:
        gripper: Gripper instance
        object_pos: Object position (numpy array)
        approach_vertex_bounds: Bounds for approach vertex (x, y, z) as tuple of tuples
        grasp_offset_bounds: Bounds for grasp offset (x, y, z) as tuple of tuples
        maxiter: Maximum number of iterations for the optimizer (default: 100)
        seed: Random seed for reproducibility
        tol: Relative tolerance for convergence (default: 0.01)
    
    Returns:
        dict: Best grasp configuration with approach_vertex, grasp_offset, prediction, probability
    """
    # Verify model can be loaded before starting optimisation
    try:
        test_model, test_scaler = loadModel()
        del test_model, test_scaler
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first using train_grasp_model.py")
        return None
    
    # Bounds for approach vertex and grasp offset
    bounds = list(approach_vertex_bounds) + list(grasp_offset_bounds)
    
    # Arguments for objective function
    objective_args = (object_pos,)
    
    # Use differential evolution for global optimisation
    result = differential_evolution(
        _objective_function,
        bounds=bounds,
        args=objective_args,
        maxiter=maxiter,
        seed=seed,
        tol=tol,  # Convergence tolerance
        polish=True,  # Polish result with local optimisation
        workers=6,  # Parallel processing enabled
        updating='deferred',  # Better for multiprocessing
        atol=0,  # Absolute tolerance
        mutation=(0.5, 1),  # Mutation factor range
        recombination=0.7,  # Crossover probability
        strategy='best1bin'  # Strategy for differential evolution
    )
    
    if not result.success:
        print(f"Warning: Optimisation did not converge. Message: {result.message}")
    
    # Extract best solution
    best_x = result.x
    best_approach_vertex = np.array(best_x[:3])
    best_grasp_offset = np.array(best_x[3:6])
    best_success_prob = -result.fun  # Negate because optimisation minimises negative probability
    
    # Get full prediction details for the best solution
    prediction, probability = predictGraspFromGripperObject(gripper, object_pos, best_approach_vertex, best_grasp_offset)
    
    if prediction is None or probability is None:
        print("Warning: Could not get prediction details for best solution.")
        return None
    
    best_grasp = {
        "approach_vertex": best_approach_vertex,
        "grasp_offset": best_grasp_offset,
        "prediction": prediction,
        "probability": probability,
        "success_probability": best_success_prob,
        "optimization_success": result.success,
        "n_iterations": result.nit
    }
    
    return best_grasp

def testGraspPrediction():
    from Grippers.TwoFingerGripper import TwoFingerGripper
    
    gripper = TwoFingerGripper()
    object_pos = np.array([1, 0, 0.06])
    approach_vertex = np.array([0, 0, 0.6])
    grasp_offset = np.array([0, 0, 0.01])
    
    prediction, probability = predictGraspFromGripperObject(gripper, object_pos, approach_vertex, grasp_offset)

    print("Prediction: ", prediction)
    print("Probability: ", probability)
    
    if prediction is not None and probability is not None:
        print(f"Prediction: {'Success' if prediction == 1 else 'Failure'}")
        print(f"Success Probability: {probability[1]:.4f}")
        print(f"Failure Probability: {probability[0]:.4f}")
    else:
        print("Could not make prediction. Please train the model first.")

def testFindBestGrasp():
    from Grippers.TwoFingerGripper import TwoFingerGripper

    gripper = TwoFingerGripper()
    object_pos = np.array([1, 0, 0.06])
    
    # Use optimiser to find best grasp
    print("Optimising for best grasp...")
    best_grasp = findBestGrasp(
        gripper, 
        object_pos,
        approach_vertex_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.3, 1.0)),
        grasp_offset_bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.1)),
        maxiter=100,
        tol=0.01
    )
    
    if best_grasp:
        print(f"\nBest Grasp Found:")
        print(f"  Approach Vertex: {best_grasp['approach_vertex']}")
        print(f"  Grasp Offset: {best_grasp['grasp_offset']}")
        print(f"  Success Probability: {best_grasp['success_probability']:.4f}")
        print(f"  Prediction: {'Success' if best_grasp['prediction'] == 1 else 'Failure'}")
        print(f"  Optimization converged: {best_grasp['optimization_success']}")
        print(f"  Iterations: {best_grasp['n_iterations']}")
    else:
        print("Could not find best grasp. Please train the model first.")

if __name__ == "__main__":
    setupEnvironment(gui=False)
    testGraspPrediction()
    #testFindBestGrasp()

