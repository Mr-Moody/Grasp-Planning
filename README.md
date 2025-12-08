# Grasp Planning
Group 7 - Thomas Moody, Andrew Lau

A machine learning grasp planning system that generates training data through physics simulation, trains a classifier to predict successful grasps, and uses optimisation to find optimal grasp configurations for robotic manipulation.

## Overview

This system provides a complete pipeline for grasp planning:

1. **Data Generation**: Simulates grasp attempts using PyBullet to generate labeled training data.
2. **Classifier Training**: Trains a Random Forest classifier to predict grasp success based on features like gripper orientation, approach direction, and grasp offset.
3. **Grasp Planning**: Uses optimisation to find the best grasp configuration for a given object position.
4. **Visualisation**: Visualises grasp success/failure data in 3D space.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system is controlled through `main.py` with different modes. All commands use the format:

```bash
python main.py --mode <mode_name> [options]
```

### Modes

#### 1. Generate Data (`generate_data`)

Generates grasp dataset by sampling different grasp configurations in a physics simulation.

**Arguments:**
- `--num_samples` (int, default: 150): Number of samples to generate
- `--gripper_type` (str, default: "TwoFingerGripper"): Type of gripper
  - Options: `TwoFingerGripper`, `RoboticArm`
- `--object_type` (str, default: "Box"): Type of object to grasp
  - Options: `Box`, `Cylinder`, `Duck`
- `--gui`: Enable GUI visualisation during simulation

**Examples:**
```bash
# Generate data with default settings (TwoFingerGripper, Box, 150 samples)
python main.py --mode generate_data

# Generate data with 150 samples for a cylinder using RoboticArm
python main.py --mode generate_data --num_samples 150 --gripper_type RoboticArm --object_type Cylinder

# Generate data with GUI visualisation enabled
python main.py --mode generate_data --gripper_type TwoFingerGripper --object_type Box --gui
```

#### 2. Train Classifier (`train_classifier`)

Trains a Random Forest classifier on the generated grasp data.

**Arguments:**
- `--data_file` (str, optional): Path to CSV data file
  - If not specified, uses the most recent file in the `Samples/` directory

**Examples:**
```bash
# Train on the most recent data file
python main.py --mode train_classifier

# Train on a specific data file
python main.py --mode train_classifier --data_file Samples/TwoFingerGripper_Box_20251208_150831.csv
```

#### 3. Test Planner (`test_planner`)

Tests the planner by finding the best grasp configuration for a given object position using optimisation.

**Arguments:**
- `--object_x` (float, default: 1.0): X position of the object
- `--object_y` (float, default: 0.0): Y position of the object
- `--object_z` (float, default: 0.06): Z position of the object
- `--maxiter` (int, default: 100): Maximum number of iterations for optimisation
- `--seed` (int, default: 42): Random seed for optimisation reproducibility
- `--tol` (float, default: 0.01): Convergence tolerance for optimisation
- `--approach_bounds` (str, default: "(-1.0,1.0),(-1.0,1.0),(0.3,1.0)"): Bounds for approach vertex
  - Format: `"(min_x,max_x),(min_y,max_y),(min_z,max_z)"`
- `--offset_bounds` (str, default: "(-0.1,0.1),(-0.1,0.1),(-0.05,0.1)"): Bounds for grasp offset
  - Format: `"(min_x,max_x),(min_y,max_y),(min_z,max_z)"`

**Examples:**
```bash
# Find best grasp with default object position
python main.py --mode test_planner

# Find best grasp for object at specific position
python main.py --mode test_planner --object_x 1.0 --object_y 0.0 --object_z 0.06

# Test with custom optimization parameters
python main.py --mode test_planner --object_x 0.5 --object_y 0.5 --object_z 0.1 --maxiter 200 --tol 0.005

```

#### 4. Visualise (`visualise`)

Visualises grasp success/failure data from a CSV file in 3D space.

**Arguments:**
- `--csv_file` (str, required): Path to CSV file to visualise
  - Can be a relative path (will check in `Samples/` directory) or absolute path

**Examples:**
```bash
# Visualise a specific data file
python main.py --mode visualise --csv_file Samples/TwoFingerGripper_Box_20251208_150831.csv

# Visualise using just filename (searches in Samples directory)
python main.py --mode visualise --csv_file TwoFingerGripper_Box_20251208_150831.csv
```

## Examples

Here's a typical workflow to go from data generation to visualisation:

```bash
# Step 1: Generate training data
python main.py --mode generate_data --num_samples 150 --gripper_type TwoFingerGripper --object_type Box

# Step 2: Train the classifier
python main.py --mode train_classifier

# Step 3: Test the planner to find best grasp
python main.py --mode test_planner --object_x 1.0 --object_y 0.0 --object_z 0.06

# Step 4: Visualise the generated data
python main.py --mode visualise --csv_file Samples/TwoFingerGripper_Box_20251208_150831.csv
```

## Output Files

- **Training Data**: Saved in `Samples/` directory as CSV files with timestamps
  - Format: `{GripperType}_{ObjectType}_{Timestamp}.csv`
- **Trained Model**: Saved as `grasp_model.pkl` and `grasp_scaler.pkl` in the project root

## Dependencies

See `requirements.txt` for the complete list. Main dependencies include:
- pybullet (physics simulation)
- numpy, pandas (data processing)
- scikit-learn (machine learning)
- scipy (optimization)
- matplotlib (visualization)

## Notes

- The trained model files (`grasp_model.pkl` and `grasp_scaler.pkl`) must exist before running `test_planner` mode
- Data files are automatically saved to the `Samples/` directory during data generation
- GUI mode is useful for debugging but significantly slows down simulation
- The system uses a Random Forest classifier with hyperparameter optimization via RandomizedSearchCV

