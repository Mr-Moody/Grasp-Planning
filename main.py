"""
Main entry point for the Grasp Planning system.

Supports four modes:
1. generate_data - Generate grasp dataset by sampling
2. train_classifier - Train a classifier on the generated data
3. test_planner - Test the planner by finding best grasps
4. visualise - Visualise grasp data from CSV files
"""
import argparse
import os
import sys
from typing import Optional

import sampling
import train_grasp_model
import predict_grasp
import pybullet as p

def modeGenerateData(args):
    """
    Generate grasp dataset by sampling.
    """

    print("=" * 60)
    print("Grasp Data Generation")
    print("=" * 60)
    
    gripper_type = args.gripper_type
    object_type = args.object_type
    num_samples = args.num_samples
    gui = args.gui
    
    print(f"Gripper Type: {gripper_type}")
    print(f"Object Type: {object_type}")
    print(f"Number of Samples: {num_samples}")
    print(f"GUI: {gui}")
    print()
    
    try:
        sampling.main(num_samples=num_samples, gripper_type=gripper_type, object_type=object_type, gui=gui)
        print("\nData generation completed successfully!")

    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)

    finally:
        # Ensure PyBullet is disconnected
        try:
            p.disconnect()
        except:
            pass


def modeTrainClassifier(args):
    """
    Train a classifier on the generated data.
    """

    print("=" * 60)
    print("Classifier Training")
    print("=" * 60)
    
    data_file = args.data_file
    model_type = args.model_type
    
    data_file_display = data_file if data_file else "Most recent file in Samples/"
    print(f"Data File: {data_file_display}")
    print(f"Model Type: {model_type}")
    print()
    
    try:
        train_grasp_model.main(sample_data_file=data_file, model_type=model_type)
        print("\nClassifier training completed successfully!")
    except Exception as e:
        print(f"Error during classifier training: {e}")
        sys.exit(1)


def parseBounds(bounds_str):
    """
    Parse bounds string like \"(-1.0,1.0),(-1.0,1.0),(0.3,1.0)\" into tuple of tuples.
    """
    try:
        # Split by \"),(\" pattern
        parts = bounds_str.split("),(")
        bounds = []

        for i, part in enumerate(parts):
            # Remove parentheses from first and last parts
            if i == 0:
                part = part.lstrip("(")
            if i == len(parts) - 1:
                part = part.rstrip(")")

            # Parse values
            values = [float(x.strip()) for x in part.split(",")]

            if len(values) != 2:
                raise ValueError("Each bound must have exactly 2 values")

            bounds.append(tuple(values))

        return tuple(bounds)

    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bounds format: {e}")


def modeTestPlanner(args):
    """
    Test the planner by finding best grasps.
    """

    print("=" * 60)
    print("Grasp Planner Testing")
    print("=" * 60)
    
    from Grippers.TwoFingerGripper import TwoFingerGripper
    import numpy as np
    
    object_pos = np.array([args.object_x, args.object_y, args.object_z])
    maxiter = args.maxiter
    gui = args.gui
    
    print(f"Object Position: {object_pos}")
    print(f"Max Iterations: {maxiter}")
    print(f"GUI: {gui}")
    print()
    
    try:
        # Setup environment
        from util import setupEnvironment
        setupEnvironment(gui=gui)
        
        # Create gripper
        gripper = TwoFingerGripper()
        
        # Parse bounds
        approach_bounds = parseBounds(args.approach_bounds)
        offset_bounds = parseBounds(args.offset_bounds)
        
        # Find best grasp
        print("Optimising for best grasp...")
        best_grasp = predict_grasp.findBestGrasp(
            gripper,
            object_pos,
            approach_vertex_bounds=approach_bounds,
            grasp_offset_bounds=offset_bounds,
            maxiter=maxiter,
            seed=args.seed,
            tol=args.tol
        )
        
        if best_grasp:
            approach_vertex = best_grasp["approach_vertex"]
            grasp_offset = best_grasp["grasp_offset"]
            success_prob = best_grasp["success_probability"]
            prediction = best_grasp["prediction"]
            opt_success = best_grasp["optimization_success"]
            n_iter = best_grasp["n_iterations"]
            prediction_text = "Success" if prediction == 1 else "Failure"
            
            print(f"\nBest Grasp Found:")
            print(f"  Approach Vertex: {approach_vertex}")
            print(f"  Grasp Offset: {grasp_offset}")
            print(f"  Success Probability: {success_prob:.4f}")
            print(f"  Prediction: {prediction_text}")
            print(f"  Optimisation converged: {opt_success}")
            print(f"  Iterations: {n_iter}")
        else:
            print("Could not find best grasp. Please train the model first.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during planner testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Ensure PyBullet is disconnected
        try:
            p.disconnect()
        except:
            pass


def modeVisualise(args):
    """
    Visualise grasp data from CSV files.
    """

    print("=" * 60)
    print("Grasp Data Visualisation")
    print("=" * 60)
    
    csv_file = args.csv_file
    
    # If relative path, check in Samples directory
    if not os.path.isabs(csv_file):
        samples_path = os.path.join("Samples", csv_file)

        if os.path.exists(samples_path):
            csv_file = samples_path

        elif not os.path.exists(csv_file):
            # Try just the filename in Samples directory
            filename = os.path.basename(csv_file)
            samples_path = os.path.join("Samples", filename)

            if os.path.exists(samples_path):
                csv_file = samples_path
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print(f"Please provide a valid path to a CSV file in the Samples directory.")
        sys.exit(1)
    
    print(f"CSV File: {csv_file}")
    print()
    
    try:
        from visualisation import visualiseGrasps
        visualiseGrasps(csv_file)
        print("\nVisualisation completed successfully!")
        
    except Exception as e:
        print(f"Error during visualisation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Grasp Planning System - Dataset generation, classifier training, and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with 150 samples
  python main.py --mode generate_data --num_samples 150 --gripper_type TwoFingerGripper --object_type Box
  
  # Train classifier on most recent data file
  python main.py --mode train_classifier
  
  # Train classifier on specific data file
  python main.py --mode train_classifier --data_file Samples/TwoFingerGripper_Box_20251208_150831.csv
  
  # Train classifier with SVM model
  python main.py --mode train_classifier --model_type SVM
  
  # Test planner to find best grasp
  python main.py --mode test_planner --object_x 1.0 --object_y 0.0 --object_z 0.06
  
  # Visualise grasp data
  python main.py --mode visualise --csv_file Samples/TwoFingerGripper_Box_20251208_150831.csv
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate_data", "train_classifier", "test_planner", "visualise"],
        help="Mode to run: generate_data, train_classifier, test_planner, or visualise"
    )
    
    # Common arguments
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI visualisation (for generate_data and test_planner modes)"
    )
    
    # Arguments for generate_data mode
    parser.add_argument(
        "--num_samples",
        type=int,
        default=150,
        help="Number of samples to generate (default: 150). Note: For TwoFingerGripper, this may not directly control the number of samples."
    )
    
    parser.add_argument(
        "--gripper_type",
        type=str,
        default="TwoFingerGripper",
        choices=["TwoFingerGripper", "RoboticArm"],
        help="Type of gripper to use (default: TwoFingerGripper)"
    )
    
    parser.add_argument(
        "--object_type",
        type=str,
        default="Box",
        choices=["Box", "Cylinder", "Duck"],
        help="Type of object to grasp (default: Box)"
    )
    
    # Arguments for train_classifier mode
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to CSV data file. If not specified, uses most recent file in Samples/ directory."
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "SVM", "LogisticRegression"],
        help="Type of classifier model to train (default: RandomForest)"
    )
    
    # Arguments for test_planner mode
    parser.add_argument(
        "--object_x",
        type=float,
        default=1.0,
        help="X position of object (default: 1.0)"
    )
    
    parser.add_argument(
        "--object_y",
        type=float,
        default=0.0,
        help="Y position of object (default: 0.0)"
    )
    
    parser.add_argument(
        "--object_z",
        type=float,
        default=0.06,
        help="Z position of object (default: 0.06)"
    )
    
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum number of iterations for optimisation (default: 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for optimisation (default: 42)"
    )
    
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Convergence tolerance for optimisation (default: 0.01)"
    )
    
    parser.add_argument(
        "--approach_bounds",
        type=str,
        default="(-1.0,1.0),(-1.0,1.0),(0.3,1.0)",
        help="Bounds for approach vertex as string: \"(-1.0,1.0),(-1.0,1.0),(0.3,1.0)\" (default: (-1.0,1.0),(-1.0,1.0),(0.3,1.0))"
    )
    
    parser.add_argument(
        "--offset_bounds",
        type=str,
        default="(-0.1,0.1),(-0.1,0.1),(-0.05,0.1)",
        help="Bounds for grasp offset as string: \"(-0.1,0.1),(-0.1,0.1),(-0.05,0.1)\" (default: (-0.1,0.1),(-0.1,0.1),(-0.05,0.1))"
    )
    
    # Arguments for visualise mode
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to CSV file to visualise (required for visualise mode)"
    )
    
    args = parser.parse_args()
    
    # Validate mode-specific required arguments
    if args.mode == "visualise" and args.csv_file is None:
        parser.error("--csv_file is required for visualise mode")
    
    # Call the handler for the mode
    if args.mode == "generate_data":
        modeGenerateData(args)
    elif args.mode == "train_classifier":
        modeTrainClassifier(args)
    elif args.mode == "test_planner":
        modeTestPlanner(args)
    elif args.mode == "visualise":
        modeVisualise(args)

if __name__ == "__main__":
    main()