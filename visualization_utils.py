"""
Logging utilities for lymphoma classification project.

This module provides formatted printing functions for:
- Dataset information and statistics
- Train/test split details
- Cross-validation progress
- Model evaluation results
- Feature importance analysis
"""

import numpy as np
from collections import Counter


def print_separator(title="", width=80):
    """
    Print a formatted section separator.
    
    Args:
        title: Optional section title to center in the separator
        width: Total width of the separator line
    """
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)


def print_dataset_info(X, y, p_ids, feat):
    """
    Print comprehensive dataset statistics.
    
    Args:
        X: Feature matrix (DataFrame or array)
        y: Target labels
        p_ids: Patient IDs
        feat: List of feature names
    """
    print_separator("DATASET INFORMATION")
    
    unique_patients = np.unique(p_ids)
    n_patients = len(unique_patients)
    n_vois = len(y)
    n_features = len(feat)
    
    print(f"Total patients: {n_patients}")
    print(f"Total VOIs (instances): {n_vois}")
    print(f"Average VOIs per patient: {n_vois/n_patients:.2f}")
    print(f"Number of features: {n_features}")
    
    # VOI-level class distribution
    n_positive = np.sum(y)
    n_negative = len(y) - n_positive
    print(f"\nClass distribution (VOI level):")
    print(f"  HL (positive): {n_positive} ({100*n_positive/n_vois:.1f}%)")
    print(f"  Others (negative): {n_negative} ({100*n_negative/n_vois:.1f}%)")
    
    # Patient-level class distribution
    patient_labels = {}
    for pid, label in zip(p_ids, y):
        if pid not in patient_labels:
            patient_labels[pid] = label
    
    pat_pos = sum(patient_labels.values())
    pat_neg = n_patients - pat_pos
    print(f"\nClass distribution (patient level):")
    print(f"  HL (positive): {pat_pos} ({100*pat_pos/n_patients:.1f}%)")
    print(f"  Others (negative): {pat_neg} ({100*pat_neg/n_patients:.1f}%)")
    
    # VOIs per patient statistics
    vois_per_patient = Counter(p_ids)
    voi_counts = list(vois_per_patient.values())
    print(f"\nVOIs per patient statistics:")
    print(f"  Min: {min(voi_counts)}")
    print(f"  Max: {max(voi_counts)}")
    print(f"  Median: {np.median(voi_counts):.1f}")
    print(f"  Mean: {np.mean(voi_counts):.2f}")
    print(f"  Std: {np.std(voi_counts):.2f}")


def print_split_info(Xtr, ytr, ptr, Xte, yte, pte):
    """
    Print train/test split information.
    
    Args:
        Xtr, ytr, ptr: Training features, labels, and patient IDs
        Xte, yte, pte: Test features, labels, and patient IDs
    """
    print_separator("TRAIN/TEST SPLIT")
    
    n_train_pat = len(np.unique(ptr))
    n_test_pat = len(np.unique(pte))
    n_train_voi = len(ytr)
    n_test_voi = len(yte)
    
    # Training set info
    train_pat_positive = np.sum([ytr[ptr==p][0] for p in np.unique(ptr)])
    print(f"Training set:")
    print(f"  Patients: {n_train_pat} ({train_pat_positive} positive, {n_train_pat-train_pat_positive} negative)")
    print(f"  VOIs: {n_train_voi} ({np.sum(ytr)} positive, {n_train_voi-np.sum(ytr)} negative)")
    print(f"  Average VOIs per patient: {n_train_voi/n_train_pat:.2f}")
    
    # Test set info
    test_pat_positive = np.sum([yte[pte==p][0] for p in np.unique(pte)])
    print(f"\nTest set:")
    print(f"  Patients: {n_test_pat} ({test_pat_positive} positive, {n_test_pat-test_pat_positive} negative)")
    print(f"  VOIs: {n_test_voi} ({np.sum(yte)} positive, {n_test_voi-np.sum(yte)} negative)")
    print(f"  Average VOIs per patient: {n_test_voi/n_test_pat:.2f}")
    
    # Split ratio
    total_pat = n_train_pat + n_test_pat
    print(f"\nSplit ratio: {n_train_pat/total_pat:.1%} train / {n_test_pat/total_pat:.1%} test")


def print_cv_header(approach_name, n_splits, random_state):
    """
    Print cross-validation configuration header.
    
    Args:
        approach_name: Name of the approach (e.g., "Instance-Space")
        n_splits: Number of CV folds
        random_state: Random seed used
    """
    print_separator(f"CROSS-VALIDATION: {approach_name}")
    print(f"Number of folds: {n_splits}")
    print(f"Stratification: By patient with group constraints")
    print(f"Random state: {random_state}")


def print_evaluation_results(approach_name, voi_metrics, pat_metrics):
    """
    Print detailed evaluation results.
    
    Args:
        approach_name: Name of the evaluated approach
        voi_metrics: Tuple of (accuracy, f1) at VOI level (or None)
        pat_metrics: Tuple of (accuracy, f1) at patient level (or None)
    """
    print_separator(f"EVALUATION RESULTS: {approach_name}")
    
    if voi_metrics is not None:
        voi_acc, voi_f1 = voi_metrics
        print(f"VOI-level metrics:")
        print(f"  Accuracy: {voi_acc:.4f}")
        print(f"  F1-score: {voi_f1:.4f}")
    
    if pat_metrics is not None:
        pat_acc, pat_f1 = pat_metrics
        print(f"\nPatient-level metrics:")
        print(f"  Accuracy: {pat_acc:.4f}")
        print(f"  F1-score: {pat_f1:.4f}")


def print_shap_results(top_features, k=5):
    """
    Print SHAP feature importance results.
    
    Args:
        top_features: List of top k feature names
        k: Number of features to display
    """
    print_separator("SHAP FEATURE IMPORTANCE")
    print(f"Top {k} most important features:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")
    print("\nSHAP plots saved:")
    print("  - results/IS_summary.png: Feature impact visualization")
    print("  - results/IS_bar.png: Feature importance ranking")


def print_final_summary(results):
    """
    Print comprehensive final summary of all experiments.
    
    Args:
        results: Dictionary containing all experimental results
    """
    print_separator("FINAL SUMMARY")
    
    # Instance-Space with all features
    print("Instance-Space (IS) - All features:")
    print(f"  VOI accuracy:     {results['IS']['voi_acc']:.4f}")
    print(f"  VOI F1-score:     {results['IS']['voi_f1']:.4f}")
    print(f"  Patient accuracy: {results['IS']['pat_acc']:.4f}")
    print(f"  Patient F1-score: {results['IS']['pat_f1']:.4f}")
    
    # Instance-Space with top features
    n_top_features = len(results['IS_top5']['features'])
    print(f"\nInstance-Space (IS) - Top {n_top_features} features:")
    print(f"  Features: {', '.join(results['IS_top5']['features'])}")
    print(f"  VOI accuracy:     {results['IS_top5']['voi_acc']:.4f}")
    print(f"  VOI F1-score:     {results['IS_top5']['voi_f1']:.4f}")
    print(f"  Patient accuracy: {results['IS_top5']['pat_acc']:.4f}")
    print(f"  Patient F1-score: {results['IS_top5']['pat_f1']:.4f}")
    
    # Embedded-Space
    print("\nEmbedded-Space (ES):")
    print(f"  Patient accuracy: {results['ES']['pat_acc']:.4f}")
    print(f"  Patient F1-score: {results['ES']['pat_f1']:.4f}")
    
    # Best performer
    print("\nPerformance comparison (patient-level F1-score):")
    f1_scores = {
        "Instance-Space (All features)": results['IS']['pat_f1'],
        f"Instance-Space (Top {n_top_features})": results['IS_top5']['pat_f1'],
        "Embedded-Space": results['ES']['pat_f1']
    }
    
    for method, f1 in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- BEST" if f1 == max(f1_scores.values()) else ""
        print(f"  {method:35s}: {f1:.4f}{marker}")
    
    print_separator()




def print_completion_message():
    """Print completion message with output file locations."""
    print("\nExperiment completed successfully!")
    print("\nResults saved to 'results/' directory:")
    print("  - summary.json: Numerical results in JSON format")
    print("  - IS_summary.png: SHAP summary plot")
    print("  - IS_bar.png: SHAP feature importance bar plot")
    print_separator()