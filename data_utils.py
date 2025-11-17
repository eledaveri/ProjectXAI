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

def merge_datasets_with_demographics(path_a, path_b, strategy="common_patients"):
    """
    Merge dataset_A and dataset_B handling missing demographics.
    
    Strategies:
    - "common_patients": Keep only patients present in both datasets (RECOMMENDED)
    - "fill_missing": Fill missing demographics with mean/mode
    - "drop_demographics": Remove sex/age from both datasets
    
    Args:
        path_a: Path to dataset_A.csv (with sex/age)
        path_b: Path to dataset_B.csv (without sex/age)
        strategy: Merging strategy
    
    Returns:
        X_combined, y_combined, patient_ids_combined, feature_cols
    """
    print(f"\nMerging datasets using strategy: '{strategy}'")
    
    # Load raw dataframes
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    
    print(f"Dataset A: {len(df_a)} rows, {len(df_a['patient'].unique())} patients")
    print(f"Dataset B: {len(df_b)} rows, {len(df_b['patient'].unique())} patients")
    
    # Check demographics presence
    has_sex_a = "sex" in df_a.columns
    has_age_a = "age" in df_a.columns
    has_sex_b = "sex" in df_b.columns
    has_age_b = "age" in df_b.columns
    
    print(f"Demographics in A: sex={has_sex_a}, age={has_age_a}")
    print(f"Demographics in B: sex={has_sex_b}, age={has_age_b}")
    
    # Strategy 1: Keep only common patients
    if strategy == "common_patients":
        patients_a = set(df_a["patient"].unique())
        patients_b = set(df_b["patient"].unique())
        common_patients = patients_a & patients_b
        
        print(f"\nCommon patients: {len(common_patients)}")
        print(f"Only in A: {len(patients_a - patients_b)}")
        print(f"Only in B: {len(patients_b - patients_a)}")
        
        if len(common_patients) == 0:
            raise ValueError("No common patients found! Cannot merge datasets.")
        
        # Filter to common patients
        df_a_filtered = df_a[df_a["patient"].isin(common_patients)].copy()
        df_b_filtered = df_b[df_b["patient"].isin(common_patients)].copy()
        
        print(f"After filtering:")
        print(f"  Dataset A: {len(df_a_filtered)} rows, {len(df_a_filtered['patient'].unique())} patients")
        print(f"  Dataset B: {len(df_b_filtered)} rows, {len(df_b_filtered['patient'].unique())} patients")
        
        # Add demographics from A to B based on patient ID
        if has_sex_a and not has_sex_b:
            # Create patient -> sex mapping from A
            patient_sex = df_a_filtered.groupby("patient")["sex"].first().to_dict()
            df_b_filtered["sex"] = df_b_filtered["patient"].map(patient_sex)
            print("  Added 'sex' to dataset B from dataset A")
        
        if has_age_a and not has_age_b:
            # Create patient -> age mapping from A
            patient_age = df_a_filtered.groupby("patient")["age"].first().to_dict()
            df_b_filtered["age"] = df_b_filtered["patient"].map(patient_age)
            print("  Added 'age' to dataset B from dataset A")
        
        # Concatenate
        df_merged = pd.concat([df_a_filtered, df_b_filtered], axis=0, ignore_index=True)
    
    # Strategy 2: Fill missing demographics
    elif strategy == "fill_missing":
        print("\nFilling missing demographics with mean/mode...")
        
        if has_sex_a and not has_sex_b:
            # Fill with mode from A
            mode_sex = df_a["sex"].mode()[0] if len(df_a["sex"].mode()) > 0 else "M"
            df_b["sex"] = mode_sex
            print(f"  Filled 'sex' in B with mode: {mode_sex}")
        
        if has_age_a and not has_age_b:
            # Fill with mean from A
            mean_age = df_a["age"].mean()
            df_b["age"] = mean_age
            print(f"  Filled 'age' in B with mean: {mean_age:.1f}")
        
        # Concatenate
        df_merged = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    
    # Strategy 3: Drop demographics from both
    elif strategy == "drop_demographics":
        print("\nDropping demographics from both datasets...")
        
        cols_to_drop = []
        if "sex" in df_a.columns:
            cols_to_drop.append("sex")
        if "age" in df_a.columns:
            cols_to_drop.append("age")
        
        df_a_no_demo = df_a.drop(columns=[c for c in cols_to_drop if c in df_a.columns], errors="ignore")
        df_b_no_demo = df_b.drop(columns=[c for c in cols_to_drop if c in df_b.columns], errors="ignore")
        
        print(f"  Dropped columns: {cols_to_drop}")
        
        # Concatenate
        df_merged = pd.concat([df_a_no_demo, df_b_no_demo], axis=0, ignore_index=True)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"\nMerged dataset: {len(df_merged)} rows, {len(df_merged['patient'].unique())} patients")
    
    # Now process the merged dataframe using standard load_data logic
    # Encode sex if present
    if "sex" in df_merged.columns:
        df_merged["sex"] = df_merged["sex"].map({"M": 0, "F": 1})
    
    # Target -> binary HL vs Others
    df_merged["target"] = df_merged["type"].apply(lambda x: 1 if x == "HL" else 0)
    
    # Feature list
    drop_cols = ["type", "target", "VOI", "patient", "weight", "height"]
    feature_cols = [c for c in df_merged.columns if c not in drop_cols]
    
    X_combined = df_merged[feature_cols].copy()
    y_combined = df_merged["target"].values
    patient_ids_combined = df_merged["patient"].values
    
    return X_combined, y_combined, patient_ids_combined, feature_cols