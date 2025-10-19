import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["sex"] = df["sex"].map({"M": 0, "F": 1})

    print("Distribuzione delle classi nel dataset:")
    print(df["type"].value_counts())

    print("Numero pazienti per tipo:")
    print(df.groupby("type")["patient"].nunique())

    # Separazione variabili
    X = df.drop(columns=["type"])
    y = df["type"]
    groups = df["patient"]

    # Encoding del target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Valori unici nel target (y_encoded):", np.unique(y_encoded))
    print("Numero di classi:", len(np.unique(y_encoded)))

    # Split pazienti in train/test
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
