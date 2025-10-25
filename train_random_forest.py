import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carica il dataset
df = pd.read_csv("dataset_A.csv")
df["sex"] = df["sex"].map({"M": 0, "F": 1})
print(df.groupby("type")["patient"].nunique())


# Colonne
X = df.drop(columns=["type"])
y = df["type"]
groups = df["patient"]  # ogni paziente ha più righe

# Suddividi i pazienti (non le righe) in train/test
unique_patients = df["patient"].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=42, stratify=df.groupby("patient")["type"].first()
)

train_idx = df["patient"].isin(train_patients)
test_idx = df["patient"].isin(test_patients)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Cross-validation rispettando i gruppi
gkf = GroupKFold(n_splits=5)

for fold, (train_index, val_index) in enumerate(gkf.split(X_train, y_train, groups=df.loc[train_idx, "patient"])):
    print(f"Fold {fold+1}")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    preds = model.predict(X_train.iloc[val_index])
    print(classification_report(y_train.iloc[val_index], preds))

# Addestra modello finale e valuta su test
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print(classification_report(y_test, y_pred))
