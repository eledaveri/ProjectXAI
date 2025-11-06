import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from joblib import dump
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

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


def aggregate_majority(y_pred, patient_ids):
    df = pd.DataFrame({"patient": patient_ids, "pred": y_pred})
    return df.groupby("patient")["pred"].agg(lambda x: x.value_counts().idxmax())


def aggregate_mean_prob(y_prob, patient_ids):
    df = pd.DataFrame({"patient": patient_ids, "prob": y_prob})
    prob_mean = df.groupby("patient")["prob"].mean()
    return (prob_mean > 0.5).astype(int)


def train_eval_IS(X_tr, y_tr, p_tr, X_te, y_te, p_te):
    model = create_xgb_binary()
    model.fit(X_tr, y_tr)

    y_pred_voi = model.predict(X_te)
    y_prob_voi = model.predict_proba(X_te)[:, 1]
    # VOI metrics
    voi_acc = accuracy_score(y_te, y_pred_voi)
    voi_f1 = f1_score(y_te, y_pred_voi)

    # PATIENT majority
    pat_pred = aggregate_majority(y_pred_voi, p_te)
    order = pd.Series(y_te, index=p_te).groupby(level=0).first()
    pat_true = order.values
    pat_pred = pat_pred.loc[order.index].values

    pat_acc = accuracy_score(pat_true, pat_pred)
    pat_f1  = f1_score(pat_true, pat_pred)

    voi_stats = eval_IS_verbose(y_te, y_pred_voi)
    pat_stats = eval_IS_verbose(pat_true, pat_pred)

    return model, (voi_acc, voi_f1), (pat_acc, pat_f1)

def cv_IS(X, y, p_ids, n_splits=3):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs = []
    f1s = []

    for tr, va in sgkf.split(X, y, groups=p_ids):
        model = create_xgb_binary()
        model.fit(X.iloc[tr], y[tr])

        y_pred = model.predict(X.iloc[va])

        # patient aggregation
        pat_pred = aggregate_majority(y_pred, p_ids[va])
        order = pd.Series(y[va], index=p_ids[va]).groupby(level=0).first()

        pat_true = order.values
        pat_pred = pat_pred.loc[order.index].values

        accs.append(accuracy_score(pat_true, pat_pred))
        f1s.append(f1_score(pat_true, pat_pred))

    print("\nCross-val IS (patient-level):")
    print(f"ACC: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

def save_model(model, path="results/model.joblib"):
    os.makedirs("results", exist_ok=True)
    dump(model, path)
    print(f"Model saved → {path}")

def eval_IS_verbose(y_true, y_pred):
    return dict(
        acc = accuracy_score(y_true, y_pred),
        f1  = f1_score(y_true, y_pred),
        prec= precision_score(y_true, y_pred),
        rec = recall_score(y_true, y_pred),
        cm  = confusion_matrix(y_true, y_pred).tolist()
    )