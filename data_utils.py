import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess_data(csv_path: str):
    """Load and preprocess the dataset from a CSV file. 
    Args:
        csv_path: Path to the CSV file.
    Returns:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        groups_train: Group labels for the training set.
        le: Fitted LabelEncoder for decoding class labels.
    """
    df = pd.read_csv(csv_path)
    df["sex"] = df["sex"].map({"M": 0, "F": 1})

    print("Number of unique patients per class:")
    print(df.groupby("type")["patient"].nunique())

    # Colonne da escludere dalla classificazione
    columns_to_exclude = ["type", "patient", "sex", "age", "weight", "height"]
    
    # Trova le colonne effettivamente presenti nel dataset
    columns_present = [col for col in columns_to_exclude if col in df.columns]
    print(f"\nColumns excluded from classification: {columns_present}")
    
    # Estrai features escludendo le colonne specificate
    X = df.drop(columns=columns_present)
    print(f"Number of features used for classification: {X.shape[1]}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    y = df["type"]
    groups = df["patient"]

    # Target encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("\nTarget unique values:", np.unique(y_encoded))
    print("Number of classes:", len(np.unique(y_encoded)))

    # Train-test split
    unique_patients = df["patient"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=42,
        stratify=df.groupby("patient")["type"].first()
    )

    train_idx = df["patient"].isin(train_patients)
    test_idx = df["patient"].isin(test_patients)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = df.loc[train_idx, "patient"]

    return X_train, X_test, y_train, y_test, groups_train, le
