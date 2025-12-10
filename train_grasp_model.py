
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import joblib
import os

from visualisation import visualiseConfusionMatrix

def loadGraspData(data_file:str="Samples/grasp_data.csv"):
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
    
    df = balanceDataset(df)
    
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

def balanceDataset(df:pd.DataFrame) -> pd.DataFrame:
    """
    Balances a dataset by downsampling the majority class to match the minority class count.
    
    Args:
        df: Input DataFrame with a label column (0/Failure, 1/Success)
    
    Returns:
        Balanced DataFrame with equal number of positive and negative samples.
    """
    # Separate positive and negative samples
    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]
    
    # Get len of smallest class
    min_count = min(len(positive), len(negative))
    
    # If missing either 1 or 0 label set then return original dataset
    if min_count == 0:
        print(f"WARNING: One class has 0 samples. Returning original dataset.")
        print(f"Original: {len(df)} samples ({len(positive)} positive, {len(negative)} negative)")
        return df
    
    if len(positive) > len(negative):
        # Downsample positive samples
        balanced_positive = positive.sample(n=min_count, random_state=42)
        balanced_negative = negative
    elif len(negative) > len(positive):
        # Downsample negative samples
        balanced_negative = negative.sample(n=min_count, random_state=42)
        balanced_positive = positive
    else:
        # Classes already balanced
        balanced_positive = positive
        balanced_negative = negative
    
    # Concatenate and shuffle
    balanced_df = pd.concat([balanced_positive, balanced_negative], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify final counts
    final_positive = len(balanced_df[balanced_df["label"] == 1])
    final_negative = len(balanced_df[balanced_df["label"] == 0])
    
    print("\nBalancing Dataset:")
    print(f"Original: {len(df)} samples ({len(positive)} positive, {len(negative)} negative)")
    print(f"Balanced: {len(balanced_df)} samples ({final_positive} positive, {final_negative} negative)")
    
    return balanced_df

def trainModel(features, labels, test_size=0.2, val_size=0.2, random_state=42, n_iter_search=150, model_type:str="RandomForest"):
    """
    Train a Random Forest or Gradient Boosting Classifier on grasp data.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label vector (n_samples,)
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation (after test split)
        random_state: Random seed for reproducibility
        n_iter_search: Number of parameter combinations to try in RandomizedSearchCV
    
    Returns:
        tuple: (model, scaler, X_test_scaled, y_test, y_test_pred, X_test, y_pred_proba)
    """
    # Check class distribution
    unique_labels = np.unique(labels)
    label_counts = np.bincount(labels)
    
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # Separate train and validation from temp
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
    else:
        # No validation set so uses all data for training
        X_train = X_temp
        y_train = y_temp
        X_val = np.array([]).reshape(0, features.shape[1])  # Empty validation set
        y_val = np.array([])  # Empty validation labels
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Only transform validation set if it's not empty
    if len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = np.array([]).reshape(0, features.shape[1])
        
    X_test_scaled = scaler.transform(X_test)

    # Check if need class_weight for imbalanced classes
    unique_labels = np.unique(labels)
    class_weight = "balanced" if len(unique_labels) > 1 else None

    if model_type == "RandomForest": 
        # More conservative base classifier to reduce overfitting
        base_clf = RandomForestClassifier(n_estimators=150, 
                                            max_depth=7,
                                            min_samples_split=20,
                                            min_samples_leaf=10,
                                            max_features="sqrt",
                                            class_weight=class_weight,
                                            bootstrap=True,
                                            random_state=42)
        
        param_distributions = {
            "n_estimators": [100, 150, 200, 250, 300],  # Reduced range, fewer trees can help
            "max_depth": [3, 5, 7, 10, 12, 15],  # Reduced max depth - shallower trees prevent overfitting
            "min_samples_split": [10, 15, 20, 25, 30, 40, 50],  # Increased - require more samples to split
            "min_samples_leaf": [4, 6, 8, 10, 12, 15, 20],  # Increased - larger leaf nodes
            "max_features": ["sqrt", "log2", 0.3, 0.4],  # Reduced - fewer features per split
            "bootstrap": [True],  # Always use bootstrap for better generalization
        }

    elif model_type == "SVM":
        base_clf = SVC(kernel = "rbf", 
                       probability=True, 
                       class_weight=class_weight, 
                       random_state=random_state)

        param_distributions = {
            "C": uniform(0.1, 100),
            "gamma": uniform(0.01, 1)
        }        

    elif model_type == "LogisticRegression":
        base_clf = LogisticRegression(class_weight=class_weight, 
                                      max_iter=5000,
                                      random_state=random_state)

        param_distributions = {
            "C": uniform(0.01, 100),
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"] 
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of: RandomForest, SVM, LogisticRegression")

    # Calculate total possible combinations (only for RandomForest, others use continuous distributions)
    if model_type == "RandomForest":
        total_combinations = (len(param_distributions["n_estimators"]) * 
                             len(param_distributions["max_depth"]) * 
                             len(param_distributions["min_samples_split"]) * 
                             len(param_distributions["min_samples_leaf"]) * 
                             len(param_distributions["max_features"]) * 
                             len(param_distributions["bootstrap"]))
        
        # Try up to n_iter_search combinations or all if fewer
        n_iter = min(n_iter_search, total_combinations)
    else:
        # For SVM and LogisticRegression, param distributions use continuous uniform distributions so use n_iter_search
        total_combinations = float("inf")
        n_iter = n_iter_search
    
    if model_type == "RandomForest":
        print(f"Total possible parameter combinations: {total_combinations}")
    else:
        print(f"Parameter space: continuous distributions")
    print(f"Exploring {n_iter} combinations")
    
    search = RandomizedSearchCV(
        base_clf, 
        param_distributions, 
        n_iter=n_iter,
        cv=10,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True  # Refit best model on full training data
    )

    print(f"Searching for best parameters...")
    search.fit(X_train_scaled, y_train)
    clf = search.best_estimator_

    print(f"\nBest parameters found: {search.best_params_}")
    best_idx = search.best_index_
    best_std = search.cv_results_["std_test_score"][best_idx] if best_idx < len(search.cv_results_["std_test_score"]) else 0.0
    print(f"Best CV score: {search.best_score_:.4f} (+/- {best_std:.4f})")

    print(f"\nTraining {model_type}...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Model is already trained by RandomizedSearchCV with refit=True
    # But we'll evaluate it on our validation set separately
    
    # Make predictions on all sets
    y_train_pred = clf.predict(X_train_scaled)
    
    if val_size > 0:
        y_val_pred = clf.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
    else:
        # No validation set
        y_val_pred = np.array([])
        val_accuracy = 0.0
        
    y_test_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)
    
    # Calculate scores for all sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "=" * 60)
    print("Model Performance Scores:")
    print("=" * 60)

    print(f"Training Accuracy:   {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Testing Accuracy:    {test_accuracy:.4f}")

    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    
    # Check for overfitting
    train_val_gap = train_accuracy - val_accuracy
    if train_val_gap > 0.15:
        print(f"\n{RED}WARNING: Large gap between train ({train_accuracy:.4f}) and validation ({val_accuracy:.4f}) accuracy!{RESET}")
        print(f"   Gap: {train_val_gap:.4f} - Model is overfitting significantly.{RESET}")
        print(f"\n   Current best parameters:")
        
        if hasattr(clf, "max_depth"):
            print(f"   - max_depth: {clf.max_depth}")
            print(f"   - min_samples_split: {clf.min_samples_split}")
            print(f"   - min_samples_leaf: {clf.min_samples_leaf}")
            print(f"   - max_features: {clf.max_features}")
            print(f"   - n_estimators: {clf.n_estimators}")
        else:
            # Print relevant parameters for other models (e.g. SVM, LogisticRegression)
            params = clf.get_params()
            for key, value in params.items():
                # Only print interesting params to avoid clutter
                if key in ["C", "gamma", "kernel", "solver", "penalty", "degree"]:
                    print(f"   - {key}: {value}")

    elif train_val_gap > 0.05:
        print(f"\n{YELLOW}Moderate overfitting detected (gap: {train_val_gap:.4f}){RESET}\n")
        
        if hasattr(clf, "max_depth"):
            print(f"   Current parameters: max_depth={clf.max_depth}, min_samples_split={clf.min_samples_split}, min_samples_leaf={clf.min_samples_leaf}")
        else:
            print("   Model parameters (subset):")
            params = clf.get_params()
            for key in ["C", "gamma", "kernel", "solver", "penalty"]:
                if key in params:
                    print(f"     {key}: {params[key]}")

    else:
        print(f"\n{GREEN}Good generalisation (train-val gap: {train_val_gap:.4f}){RESET}")
    
    print("=" * 60)
    
    # Evaluate model (keep existing output for backward compatibility)
    accuracy = test_accuracy
    print(f"\nModel Accuracy (Test): {accuracy:.4f}")
    
    # Check which classes are present in the test set
    unique_classes_test = np.unique(y_test)
    unique_classes_pred = np.unique(y_test_pred)
    all_classes = np.unique(np.concatenate([unique_classes_test, unique_classes_pred]))
    
    # Classification report
    print("\nClassification Report (Test Set):")

    if len(all_classes) == 1:
        print(f"Warning: Only one class ({'Success' if all_classes[0] == 1 else 'Failure'}) present in test set.")
        print("Cannot generate classification report with only one class.")
        print(f"All test samples are: {'Success' if all_classes[0] == 1 else 'Failure'}")
    else:
        # Use labels to make sure both classes are included even if one is missing
        print(classification_report(y_test, y_test_pred, 
                                   labels=[0, 1],
                                   target_names=["Failure", "Success"],
                                   zero_division=0.0))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    print(f"Confusion Matrix:\n{cm}")
    
    #visualiseConfusionMatrix(cm)
    
    # # Feature importance
    # print("\nFeature Importance:")

    # feature_names = [
    #     "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
    #     "Offset X", "Offset Y", "Offset Z",
    #     "Approach Dir X", "Approach Dir Y", "Approach Dir Z",
    #     "Approach Distance"
    # ]

    # importances = clf.feature_importances_
    # indices = np.argsort(importances)[::-1]
    
    # for i in indices:
    #     print(f"  {feature_names[i]}: {importances[i]:.4f}")

    return clf, scaler, X_test_scaled, y_test, y_test_pred, X_test, y_pred_proba

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

def runCrossValidation(features, labels, n_splits=3, test_size=0.2, model_type="RandomForest", random_state=42):
    """
    Run cross-validation by randomly splitting train (80%) and test (20%) sets multiple times.
    
    Args:
        features: Feature matrix
        labels: Label vector
        n_splits: Number of random splits to perform (default: 3)
        test_size: Proportion of data for testing (default: 0.2, meaning 80% train, 20% test)
        model_type: Type of classifier model to train
        random_state: Base random seed (will be incremented for each split)
    
    Returns:
        dict: Dictionary containing average performance metrics across all splits
    """
    print("=" * 60)
    print("Cross-Validation: Multiple Train/Test Splits")
    print("=" * 60)
    print(f"Number of splits: {n_splits}")
    print(f"Train/Test split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
    print(f"Model type: {model_type}\n")
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for split_idx in range(n_splits):
        print(f"Split {split_idx + 1}/{n_splits}...")
        split_seed = random_state + split_idx
        
        # Train model with this split
        model, scaler, X_test_scaled, y_test, y_test_pred, X_test, y_pred_proba = trainModel(
            features,
            labels,
            test_size=test_size,
            val_size=0.0,  # No validation set for cross-validation
            random_state=split_seed,
            model_type=model_type,
            n_iter_search=100
        )
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="binary", zero_division=0.0) if len(np.unique(y_test)) > 1 else 0.0
        recall = recall_score(y_test, y_test_pred, average="binary", zero_division=0.0) if len(np.unique(y_test)) > 1 else 0.0
        f1 = f1_score(y_test, y_test_pred, average="binary", zero_division=0.0) if len(np.unique(y_test)) > 1 else 0.0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    
    # Calculate statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print("=" * 60)
    print("Cross-Validation Results Summary")
    print("=" * 60)
    print(f"Number of splits: {n_splits}\n")
    print("Average Performance (Mean +/- Std):")
    print(f"  Accuracy:  {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
    print(f"  Precision: {mean_precision:.4f} +/- {std_precision:.4f}")
    print(f"  Recall:    {mean_recall:.4f} +/- {std_recall:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f} +/- {std_f1:.4f}\n")
    print("Individual Split Results:")
    for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
        print(f"  Split {i+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print("=" * 60)
    
    return {
        "n_splits": n_splits,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores
    }


def compareModels(features, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Compare Random Forest and Gradient Boosting models to find the best one.
    
    Args:
        features: Feature matrix
        labels: Label vector
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
    
    Returns:
        tuple: (best_model, best_scaler, best_model_type, results_dict)
    """
    print("=" * 60)
    print("Comparing Models")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test Random Forest
    print("Testing Random Forest...")
    model_rf, scaler_rf, _, _, _, _, _ = trainModel(features,
                                                    labels, 
                                                    test_size=test_size, 
                                                    val_size=val_size,
                                                    random_state=random_state)
    
    # Extract validation accuracy from the trained models
    # The trainModel function already returns validation accuracy, but we need to extract it
    # For simplicity, we'll use the validation set that trainModel creates internally
    # We'll just use the returned scalers which are already fitted
    
    # Get validation accuracy for RF (using the scaler from trainModel)
    # We need to recreate the same split to get validation set
    unique_labels = np.unique(labels)
    use_stratify = len(unique_labels) > 1
    if use_stratify:
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
    
    # Use the scaler returned from trainModel (already fitted on training data)
    X_val_scaled_rf = scaler_rf.transform(X_val)
    y_val_pred_rf = model_rf.predict(X_val_scaled_rf)
    val_acc_rf = accuracy_score(y_val, y_val_pred_rf)
    results["random_forest"] = {"model": model_rf, "scaler": scaler_rf, "val_accuracy": val_acc_rf}
    
    print(f"\nRandom Forest Validation Accuracy: {val_acc_rf:.4f}\n")
    
    # Test Gradient Boosting
    print("Testing Gradient Boosting...")
    model_gb, scaler_gb, _, _, _, _, _ = trainModel(features, labels, test_size=test_size, val_size=val_size,
                                                    random_state=random_state)
    
    # Use the scaler returned from trainModel
    X_val_scaled_gb = scaler_gb.transform(X_val)
    y_val_pred_gb = model_gb.predict(X_val_scaled_gb)
    val_acc_gb = accuracy_score(y_val, y_val_pred_gb)
    results["gradient_boosting"] = {"model": model_gb, "scaler": scaler_gb, "val_accuracy": val_acc_gb}
    
    print(f"\nGradient Boosting Validation Accuracy: {val_acc_gb:.4f}\n")
    
    # Determine best model
    if val_acc_rf > val_acc_gb:
        best_model_type = "random_forest"
        best_model = model_rf
        best_scaler = scaler_rf
    else:
        best_model_type = "gradient_boosting"
        best_model = model_gb
        best_scaler = scaler_gb
    
    print("=" * 60)
    print(f"Best Model: {best_model_type.upper()} (Val Accuracy: {results[best_model_type]["val_accuracy"]:.4f})")
    print("=" * 60)
    
    return best_model, best_scaler, best_model_type, results

def main(sample_data_file:Optional[str]=None, model_type:str="RandomForest"):
    print("=" * 60)
    print("Grasp Prediction Model Training")
    print("=" * 60 + "\n")

    print("Loading data...")

    data_file = None

    if sample_data_file is not None:
        path = os.path.join("Samples", sample_data_file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} not found.")

        data_file = path
    
    else:
        samples_dir = "Samples"
        csv_files = [f for f in os.listdir(samples_dir) if f.endswith('.csv')]
        if csv_files:
            # Use the most recent file (by modification time)
            csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(samples_dir, f)), reverse=True)
            data_file = os.path.join(samples_dir, csv_files[0])
            print(f"   Using most recent file: {csv_files[0]}")


    features, labels, object_types = loadGraspData(data_file)
    print(f"   Loaded {len(features)} samples")
    print(f"   Features shape: {features.shape}")
    print(f"   Success rate: {np.mean(labels) * 100:.2f}%")
    
    print("\nTraining model...\n")

    model, scaler, X_test, y_test, y_pred, X_test_original, y_pred_proba = trainModel(
        features, 
        labels,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        model_type=model_type)
    
    # Save model
    print("\nSaving model...")
    saveModel(model, scaler)

if __name__ == "__main__":
    main(sample_data_file="TwoFingerGripper_Box_20251208_140409.csv")

