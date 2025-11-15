import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Encode sex if present
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"M": 0, "F": 1})

    # Target -> binary HL vs Others
    df["target"] = df["type"].apply(lambda x: 1 if x == "HL" else 0)

    # Feature list
    drop_cols = ["type", "target", "VOI", "patient", "weight", "height"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_voi = df[feature_cols].copy()
    y_voi = df["target"].values
    patient_ids = df["patient"].values

    return X_voi, y_voi, patient_ids, feature_cols


def load_data_exclude_features(csv_path, exclude_cols=None):
    """
    Load data and optionally exclude demographic features.
    
    Args:
        csv_path: Path to CSV file
        exclude_cols: List of column names to exclude (e.g., ["sex", "age"])
    
    Returns:
        X_voi, y_voi, patient_ids, feature_cols
    """
    if exclude_cols is None:
        exclude_cols = []
    
    df = pd.read_csv(csv_path)

    # Encode sex if present
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"M": 0, "F": 1})

    # Target -> binary HL vs Others
    df["target"] = df["type"].apply(lambda x: 1 if x == "HL" else 0)

    # Feature list
    drop_cols = ["type", "target", "VOI", "patient", "weight", "height"] + exclude_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_voi = df[feature_cols].copy()
    y_voi = df["target"].values
    patient_ids = df["patient"].values

    return X_voi, y_voi, patient_ids, feature_cols


def split_by_patient(X, y, patient_ids, test_size, random_state):
    unique_patients = np.unique(patient_ids)

    # Unique patient → label
    tmp = pd.DataFrame({"p": patient_ids, "y": y})
    patient_y = tmp.groupby("p")["y"].first()

    train_pat, test_pat = train_test_split(
        patient_y.index, 
        test_size=test_size, 
        random_state=random_state,
        stratify=patient_y.values
    )

    mask_train = np.isin(patient_ids, train_pat)
    mask_test  = np.isin(patient_ids, test_pat)

    return (
        X[mask_train], y[mask_train], patient_ids[mask_train],
        X[mask_test],  y[mask_test],  patient_ids[mask_test]
    )


def align_features(X_source, feature_cols_source, X_target):
    """
    Align X_target columns to match X_source column order.
    
    Args:
        X_source: DataFrame with reference column order
        feature_cols_source: List of feature column names from source
        X_target: DataFrame to align
    
    Returns:
        X_target reordered and filtered to match X_source columns
    """
    # Get common features between source and target
    source_cols = list(X_source.columns)
    target_cols = list(X_target.columns)
    
    # Find features that exist in both datasets
    common_features = [col for col in source_cols if col in target_cols]
    
    # Check if we're missing important features
    missing = [col for col in source_cols if col not in target_cols]
    if missing:
        print(f"Warning: Following features missing in target dataset: {missing}")
    
    # Reorder target to match source column order
    X_target_aligned = X_target[common_features].copy()
    
    return X_target_aligned


def combine_datasets(X1, y1, p1, X2, y2, p2):
    """
    Combine two datasets, adjusting patient IDs to avoid conflicts.
    Aligns X2 columns to match X1.
    
    Args:
        X1, y1, p1: First dataset (features, labels, patient IDs)
        X2, y2, p2: Second dataset (features, labels, patient IDs)
    
    Returns:
        X_combined, y_combined, p_combined
    """
    # Align X2 columns to X1
    X2_aligned = align_features(X1, list(X1.columns), X2)
    
    # Ensure X2 has same columns as X1 (in same order)
    for col in X1.columns:
        if col not in X2_aligned.columns:
            X2_aligned[col] = np.nan
    X2_aligned = X2_aligned[X1.columns]
    
    # Adjust patient IDs from dataset_2 to avoid conflicts
    max_pid = p1.max()
    p2_adjusted = p2 + max_pid + 1
    
    # Concatenate
    X_combined = pd.concat(
        [X1.reset_index(drop=True), X2_aligned.reset_index(drop=True)], 
        ignore_index=True
    )
    y_combined = np.concatenate([y1, y2])
    p_combined = np.concatenate([p1, p2_adjusted])
    
    return X_combined, y_combined, p_combined