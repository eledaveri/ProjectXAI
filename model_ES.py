import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from joblib import dump
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

def create_patient_embedding(X, y, patient_ids):
    df = X.copy()
    df["patient"] = patient_ids
    df["y"] = y

    agg = df.groupby("patient").agg(["min", "max", "mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    y_pat = df.groupby("patient")["y"].first()

    return agg, y_pat


def create_xgb_binary():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


def train_eval_ES(Xp_tr, yp_tr, Xp_te, yp_te):
    model = create_xgb_binary()
    model.fit(Xp_tr, yp_tr)

    y_pred = model.predict(Xp_te)

    acc = accuracy_score(yp_te, y_pred)
    f1  = f1_score(yp_te, y_pred)

    return model, (acc, f1)

def cv_ES(Xp, yp, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    stats = []

    for tr, va in skf.split(Xp, yp):
        X_train, y_train = Xp.iloc[tr], yp.iloc[tr]
        X_valid, y_valid = Xp.iloc[va], yp.iloc[va]

        model = create_xgb_binary()
        model.fit(X_train, y_train)

        pred = model.predict(X_valid)

        cm = confusion_matrix(y_valid, pred)
        stats.append(dict(
            acc = accuracy_score(y_valid, pred),
            f1  = f1_score(y_valid, pred),
            prec= precision_score(y_valid, pred),
            rec = recall_score(y_valid, pred),
            cm  = cm.tolist(),
        ))

    df = pd.DataFrame(stats)

    print("\nCross-val ES:")
    print(df)
    print("\nMean:")
    print(df.mean(numeric_only=True))
    print("\nStd:")
    print(df.std(numeric_only=True))

    return df


def save_model(model, path="results/model.joblib"):
    os.makedirs("results", exist_ok=True)
    dump(model, path)
    print(f"Model saved → {path}")
