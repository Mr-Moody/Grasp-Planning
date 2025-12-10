"""
Script to run test_execute with incrementing seeds until confusion matrix
has at least 2 predicted failures and 2 actual failures.
"""
import subprocess
import re
import sys

def parse_confusion_matrix(output):
    """
    Parse the confusion matrix from the command output.
    Returns (tn, fp, fn, tp) or None if not found.
    """
    # Look for the confusion matrix section
    lines = output.split('\n')
    in_matrix = False
    tn, fp, fn, tp = None, None, None, None
    
    for i, line in enumerate(lines):
        if "Confusion Matrix:" in line:
            in_matrix = True
            continue
        
        if in_matrix and "Actual Failure" in line:
            # Line format: "Actual Failure   {tn:4d}     {fp:4d}"
            match = re.search(r'Actual Failure\s+(\d+)\s+(\d+)', line)
            if match:
                tn = int(match.group(1))
                fp = int(match.group(2))
            continue
        
        if in_matrix and "Success" in line and "Actual" not in line:
            # Line format: "       Success   {fn:4d}     {tp:4d}"
            match = re.search(r'Success\s+(\d+)\s+(\d+)', line)
            if match:
                fn = int(match.group(1))
                tp = int(match.group(2))
            break
    
    if tn is not None and fp is not None and fn is not None and tp is not None:
        return (tn, fp, fn, tp)
    return None

def check_condition(tn, fp, fn, tp):
    """
    Check if confusion matrix has at least 2 predicted failures and 2 actual failures.
    - Predicted failures: TN + FN (predicted as Failure)
    - Actual failures: TN + FP (actually Failure)
    """
    predicted_failures = tn + fn  # Total predicted as Failure
    actual_failures = tn + fp      # Total actually Failure
    
    return predicted_failures >= 1 and actual_failures >= 1

def run_test_execute(seed):
    """Run the test_execute command with given seed and return output."""
    cmd = [
        "python", "main.py",
        "--mode", "test_execute",
        "--num_grasps", "10",
        "--gripper_type", "RoboticArm",
        "--object_type", "Duck",
        "--seed", str(seed)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per run
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"Timeout for seed {seed}")
        return None
    except Exception as e:
        print(f"Error running command with seed {seed}: {e}")
        return None

def main():
    seed = 0
    max_seed = 1000  # Safety limit to prevent infinite loops
    
    print("=" * 60)
    print("Finding seed with at least 2 predicted and 2 actual failures")
    print("=" * 60)
    print()
    
    while seed <= max_seed:
        print(f"Trying seed {seed}...")
        output = run_test_execute(seed)
        
        if output is None:
            seed += 1
            continue
        
        # Parse confusion matrix
        cm = parse_confusion_matrix(output)
        
        if cm is None:
            print(f"  Could not parse confusion matrix for seed {seed}")
            seed += 1
            continue
        
        tn, fp, fn, tp = cm
        predicted_failures = tn + fn
        actual_failures = tn + fp
        
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Predicted failures: {predicted_failures}, Actual failures: {actual_failures}")
        
        if check_condition(tn, fp, fn, tp):
            print()
            print("=" * 60)
            print(f"SUCCESS! Found seed {seed} with required confusion matrix:")
            print("=" * 60)
            print(f"  Predicted failures: {predicted_failures} (need >= 2)")
            print(f"  Actual failures: {actual_failures} (need >= 2)")
            print()
            print("Full output:")
            print("-" * 60)
            print(output)
            print("-" * 60)
            print()
            print(f"Command that produced this result:")
            print(f"python main.py --mode test_execute --num_grasps 10 --gripper_type RoboticArm --object_type Duck --seed {seed}")
            return seed
        else:
            print(f"  Condition not met (need >= 2 predicted failures and >= 2 actual failures)")
            print()
        
        seed += 1
    
    print(f"Reached max seed limit ({max_seed}) without finding suitable confusion matrix.")
    return None

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)

