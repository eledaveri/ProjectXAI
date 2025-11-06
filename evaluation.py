import json
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import numpy as np


def save_report(results_dict, filename="results/summary.json"):
    """
    Save evaluation results to JSON file.
    
    Args:
        results_dict: Dictionary containing all experimental results
        filename: Output file path
    """
    os.makedirs("results", exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Summary saved to: {filename}")


def compute_all_metrics(y_true, y_pred):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, precision, recall, f1, and confusion matrix
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def print_confusion_matrix(cm, class_names=None):
    """
    Print formatted confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 array)
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  {class_names[0]:10s} {class_names[1]:10s}")
    print(f"Actual {class_names[0]:10s} {tn:10d} {fp:10d}")
    print(f"       {class_names[1]:10s} {fn:10d} {tp:10d}")
    
    # Additional statistics
    total = tn + fp + fn + tp
    print(f"\nStatistics:")
    print(f"  True Negatives:  {tn:4d} ({100*tn/total:.1f}%)")
    print(f"  False Positives: {fp:4d} ({100*fp/total:.1f}%)")
    print(f"  False Negatives: {fn:4d} ({100*fn/total:.1f}%)")
    print(f"  True Positives:  {tp:4d} ({100*tp/total:.1f}%)")


def save_detailed_report(results_dict, metrics_dict, filename="results/detailed_report.json"):
    """
    Save detailed evaluation report including all metrics.
    
    Args:
        results_dict: Main results dictionary
        metrics_dict: Additional detailed metrics
        filename: Output file path
    """
    os.makedirs("results", exist_ok=True)
    
    full_report = {
        "summary": results_dict,
        "detailed_metrics": metrics_dict
    }
    
    with open(filename, "w") as f:
        json.dump(full_report, f, indent=4)
    
    print(f"Detailed report saved to: {filename}")


def print_classification_summary(y_true, y_pred, target_names=None):
    """
    Print sklearn classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
    """
    if target_names is None:
        target_names = ["Others", "HL"]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def compare_models(results_list, model_names, metric="f1"):
    """
    Compare multiple models on a given metric.
    
    Args:
        results_list: List of result dictionaries
        model_names: List of model names
        metric: Metric to compare (default: "f1")
    """
    print(f"\nModel Comparison ({metric}):")
    print("-" * 50)
    
    scores = []
    for name, results in zip(model_names, results_list):
        score = results.get(metric, 0.0)
        scores.append((name, score))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(scores, 1):
        marker = " <-- Best" if i == 1 else ""
        print(f"  {i}. {name:30s}: {score:.4f}{marker}")