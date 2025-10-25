import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


df = pd.read_csv("dataset_A.csv")
df["sex"] = df["sex"].map({"M": 0, "F": 1})
print(df.groupby("type")["patient"].nunique())


X = df.drop(columns=["type"])
y = df["type"]
groups = df["patient"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


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


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(
    sgkf.split(X_train, y_train, groups=groups_train)
):
    print(f"\nFold {fold+1}")
    
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=y_encoded.max() + 1,
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    
    print(classification_report(
        y_val,
        preds,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0
    ))


final_model = XGBClassifier(
    objective="multi:softmax",
    num_class=y_encoded.max() + 1,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)

final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)

print("\nRisultati sul test set:")
print(classification_report(
    y_test,
    y_pred_test,
    labels=range(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))
