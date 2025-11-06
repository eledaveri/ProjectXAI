import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def explain_shap(model, X, out_prefix="results/shap"):
    """
    Compute SHAP values on training data (no leakage).
    For binary XGB: shap_values = (n_samples, n_features)
    """
    os.makedirs("results", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)

    if hasattr(shap_vals, "values"):
        vals = shap_vals.values
    else:
        vals = shap_vals

    # summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(vals, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_summary.png", dpi=300)
    plt.close()

    # bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(vals, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", dpi=300)
    plt.close()

    return vals


def top_shap_features(shap_vals, feature_names, k=10):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:k]
    return [feature_names[i] for i in idx]
