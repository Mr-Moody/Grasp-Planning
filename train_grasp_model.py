
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

def loadGraspData(data_file="Samples/grasp_data.csv"):
    """
    Load grasp data from CSV file.
    
    Args:
        data_file: Path to CSV file containing grasp data
    
    Returns:
        tuple: (features, labels, object_types) where features is a numpy array
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found. Please run sampling.py first.")
    
    # Load data from CSV using pandas
    df = pd.read_csv(data_file)
    
    feature_columns = [
        "orientation_roll",
        "orientation_pitch",
        "orientation_yaw",
        "offset_x",
        "offset_y",
        "offset_z",
        "approach_dir_x",
        "approach_dir_y",
        "approach_dir_z",
        "approach_distance"
    ]
    
    features = df[feature_columns].values
    labels = df["label"].values
    object_types = df["object_type"].values.tolist()
    
    return features, labels, object_types

def trainModel(features, labels, test_size=0.2, random_state=42, n_estimators=100, max_depth=10):
    """
    Train a Random Forest Classifier on grasp data.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label vector (n_samples,)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the random forest
        max_depth: Maximum depth of the trees
    
    Returns:
        tuple: (model, X_test, y_test, y_pred) for evaluation
    """
    # Check class distribution
    unique_labels = np.unique(labels)
    label_counts = np.bincount(labels)
    print(f"Classes in dataset: {unique_labels}")
    class_dist = dict(zip(range(len(label_counts)), label_counts))
    print(f"Class distribution: {class_dist}")
    
    # Warn if only one class is present
    if len(unique_labels) == 1:
        print(f"\nWARNING: Only one class ({'Success' if unique_labels[0] == 1 else 'Failure'}) present in dataset.")
        print("   The model will always predict this class. Consider collecting more diverse data.")
        print("   The model cannot learn to distinguish between successful and failed grasps.")
    
    # Split data into training and testing sets
    # Only use stratify if we have at least 2 samples of each class and more than one class
    use_stratify = (len(unique_labels) > 1 and 
                   all(label_counts[label] >= 2 for label in unique_labels if label < len(label_counts)))
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        print("Warning: Cannot use stratified split. Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
    
    # Optional: Scale features (Random Forest doesn't strictly need this, but it can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train Random Forest Classifier
    # Only use class_weight="balanced" if we have multiple classes
    clf_kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if len(unique_labels) > 1:
        clf_kwargs["class_weight"] = "balanced"  # Handle imbalanced classes
    
    clf = RandomForestClassifier(**clf_kwargs)
    
    print(f"Training Random Forest with {n_estimators} trees...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train the model
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Check which classes are present in the test set
    unique_classes_test = np.unique(y_test)
    unique_classes_pred = np.unique(y_pred)
    all_classes = np.unique(np.concatenate([unique_classes_test, unique_classes_pred]))
    
    # Classification report (only if we have data to report on)
    print("\nClassification Report:")
    if len(all_classes) == 1:
        print(f"Warning: Only one class ({'Success' if all_classes[0] == 1 else 'Failure'}) present in test set.")
        print("Cannot generate classification report with only one class.")
        print(f"All test samples are: {'Success' if all_classes[0] == 1 else 'Failure'}")
    else:
        # Use labels parameter to ensure both classes are included even if one is missing
        print(classification_report(y_test, y_pred, 
                                   labels=[0, 1],
                                   target_names=["Failure", "Success"],
                                   zero_division=0.0))
    
    # Confusion matrix (labels=[0,1] ensures 2x2 matrix even if classes are missing)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("      Predicted")
    print("      Failure  Success")
    print(f"True Failure    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"     Success    {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = [
        "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
        "Offset X", "Offset Y", "Offset Z",
        "Approach Dir X", "Approach Dir Y", "Approach Dir Z",
        "Approach Distance"
    ]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
    
    # Cross-validation
    print("\nCross-Validation Scores:")
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return clf, scaler, X_test_scaled, y_test, y_pred, X_test, y_pred_proba

def saveModel(model, scaler, model_file="grasp_model.pkl", scaler_file="grasp_scaler.pkl"):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        model_file: Path to save the model
        scaler_file: Path to save the scaler
    """
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"\nModel saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")

def loadModel(model_file="grasp_model.pkl", scaler_file="grasp_scaler.pkl"):
    """
    Load trained model and scaler from disk.
    
    Args:
        model_file: Path to the model file
        scaler_file: Path to the scaler file
    
    Returns:
        tuple: (model, scaler)
    """
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def predictGrasp(model, scaler, orientation_roll, orientation_pitch, orientation_yaw,
                  offset_x, offset_y, offset_z,
                  approach_dir_x, approach_dir_y, approach_dir_z,
                  approach_distance):
    """
    Predict if a grasp will be successful given the features.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        orientation_roll, orientation_pitch, orientation_yaw: Euler angles
        offset_x, offset_y, offset_z: Offset from object center
        approach_dir_x, approach_dir_y, approach_dir_z: Normalized approach direction
        approach_distance: Distance from approach point to object
    
    Returns:
        tuple: (prediction, probability) where prediction is 0 (failure) or 1 (success)
    """
    # Create feature vector
    features = np.array([[
        orientation_roll,
        orientation_pitch,
        orientation_yaw,
        offset_x,
        offset_y,
        offset_z,
        approach_dir_x,
        approach_dir_y,
        approach_dir_z,
        approach_distance
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def main():
    print("=" * 60)
    print("Grasp Prediction Model Training")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading data")
    try:
        # Try to find the most recent CSV file in Samples directory
        samples_dir = "Samples"
        if os.path.exists(samples_dir):
            csv_files = [f for f in os.listdir(samples_dir) if f.endswith('.csv')]
            if csv_files:
                # Use the most recent file (by modification time)
                csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(samples_dir, f)), reverse=True)
                data_file = os.path.join(samples_dir, csv_files[0])
                print(f"   Using most recent file: {csv_files[0]}")
            else:
                data_file = "Samples/grasp_data.csv"
        else:
            data_file = "Samples/grasp_data.csv"
        
        features, labels, object_types = loadGraspData(data_file)
        print(f"   Loaded {len(features)} samples")
        print(f"   Features shape: {features.shape}")
        print(f"   Success rate: {np.mean(labels) * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return
    
    # Train model
    print("Training model")
    model, scaler, X_test, y_test, y_pred, X_test_original, y_pred_proba = trainModel(
        features, labels,
        test_size=0.2,
        random_state=42,
        n_estimators=100,
        max_depth=10
    )
    
    # Save model
    print("Saving model")
    saveModel(model, scaler)
    
    # Example prediction
    print("Example prediction")
    print("Testing with sample features")
    example_pred, example_proba = predictGrasp(
        model, scaler,
        orientation_roll=0.0,
        orientation_pitch=1.57,  # ~90 degrees
        orientation_yaw=0.0,
        offset_x=0.0,
        offset_y=0.0,
        offset_z=0.01,  # Top of object
        approach_dir_x=0.0,
        approach_dir_y=0.0,
        approach_dir_z=-1.0,  # Approaching from above
        approach_distance=0.6
    )
    print(f"Prediction: {'Success' if example_pred == 1 else 'Failure'}")
    print(f"Probability: {example_proba[1]:.4f} (success), {example_proba[0]:.4f} (failure)")
    
    print("Training complete")

if __name__ == "__main__":
    main()

