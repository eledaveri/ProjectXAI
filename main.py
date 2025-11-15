from data_utils import (
    load_data, load_data_exclude_features, split_by_patient, 
    align_features, combine_datasets
)
from model_IS import train_eval_IS, cv_IS
from model_ES import create_patient_embedding, train_eval_ES, cv_ES
from shap_utils import explain_shap, top_shap_features
from evaluation import save_report
from visualization_utils import (
    print_dataset_info,
    print_split_info,
    print_cv_header,
    print_evaluation_results,
    print_shap_results,
    print_final_summary,
    print_completion_message,
    plot_performance_comparison
)
from config import (
    TEST_SIZE, N_SPLITS, RANDOM_STATE,
    DATASET_A_PATH, DATASET_B_PATH,
    EXCLUDE_FEATURES, TOP_K_FEATURES,
    RESULTS_DIR, SHAP_PREFIX
)
import pandas as pd
import numpy as np


def main():

    # ============================================================================
    # PHASE 1: Train on dataset_A, test on dataset_B
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 1: TRAIN ON DATASET_A, VALIDATE ON DATASET_B")
    print("="*80)
    
    # Load data
    print("\nLoading dataset_A...")
    X_A, y_A, p_ids_A, feat_A = load_data(DATASET_A_PATH)
    print_dataset_info(X_A, y_A, p_ids_A, feat_A)
    
    print("\nLoading dataset_B...")
    X_B, y_B, p_ids_B, feat_B = load_data(DATASET_B_PATH)
    print_dataset_info(X_B, y_B, p_ids_B, feat_B)
    
    # Align dataset_A to dataset_B (remove sex/age if not in B)
    print("\nAligning features: removing columns not present in dataset_B...")
    X_A_aligned = align_features(X_B, feat_B, X_A)
    feat_A_aligned = list(X_A_aligned.columns)
    print(f"Dataset_A aligned features: {len(feat_A_aligned)} (was {len(feat_A)})")
    
    # Split data_A by patient (stratified)
    print("\nSplitting dataset_A by patient (stratified)...")
    Xtr, ytr, ptr, Xte_A, yte_A, pte_A = split_by_patient(
        X_A_aligned, y_A, p_ids_A, TEST_SIZE, RANDOM_STATE
    )
    print_split_info(Xtr, ytr, ptr, Xte_A, yte_A, pte_A)
    
    # Cross-validation on dataset_A: Instance-Space
    print_cv_header("Instance-Space (IS) on dataset_A", N_SPLITS, RANDOM_STATE)
    cv_IS(X_A_aligned, y_A, p_ids_A, N_SPLITS)
    
    # Cross-validation on dataset_A: Embedded-Space
    print_cv_header("Embedded-Space (ES) on dataset_A", N_SPLITS, RANDOM_STATE)
    Xp_A, yp_A = create_patient_embedding(X_A_aligned, y_A, p_ids_A)
    cv_ES(Xp_A, yp_A, N_SPLITS)
    
    # Train and evaluate Instance-Space model on dataset_A
    print("\nTraining Instance-Space model on dataset_A train set...")
    model_IS, voi_m, pat_m = train_eval_IS(Xtr, ytr, ptr, Xte_A, yte_A, pte_A)
    print_evaluation_results("Instance-Space (All features) - A test", voi_m, pat_m)
    
    # SHAP analysis for feature importance
    print("\nComputing SHAP values for feature importance...")
    shap_vals = explain_shap(model_IS, Xtr, f"{RESULTS_DIR}/{SHAP_PREFIX}")
    top5 = top_shap_features(shap_vals, feat_A_aligned, TOP_K_FEATURES)
    print_shap_results(top5, k=TOP_K_FEATURES)
    
    # Retrain Instance-Space with top K features
    print(f"\nRetraining Instance-Space model with top {TOP_K_FEATURES} features...")
    model_IS2, voi_m2, pat_m2 = train_eval_IS(
        Xtr[top5], ytr, ptr,
        Xte_A[top5], yte_A, pte_A
    )
    print_evaluation_results(f"Instance-Space (Top {TOP_K_FEATURES} features) - A test", voi_m2, pat_m2)
    
    # Train Embedded-Space with ALL features
    print("\nTraining Embedded-Space model with all features...")
    Xptr, yptr = create_patient_embedding(Xtr, ytr, ptr)
    Xpte_A, ypte_A = create_patient_embedding(Xte_A, yte_A, pte_A)
    model_ES, pat_m_es = train_eval_ES(Xptr, yptr, Xpte_A, ypte_A)
    print_evaluation_results("Embedded-Space (All features) - A test", None, pat_m_es)
    
    # Train Embedded-Space with top K features only
    print(f"\nTraining Embedded-Space model with top {TOP_K_FEATURES} features...")
    Xptr_top5, yptr_top5 = create_patient_embedding(Xtr[top5], ytr, ptr)
    Xpte_A_top5, ypte_A_top5 = create_patient_embedding(Xte_A[top5], yte_A, pte_A)
    model_ES2, pat_m_es2 = train_eval_ES(Xptr_top5, yptr_top5, Xpte_A_top5, ypte_A_top5)
    print_evaluation_results(f"Embedded-Space (Top {TOP_K_FEATURES} features) - A test", None, pat_m_es2)
    
    # ============================================================================
    # VALIDATION ON DATASET_B
    # ============================================================================
    print("\n" + "="*80)
    print(" VALIDATION ON DATASET_B (models trained on dataset_A)")
    print("="*80)
    
    print("\nEvaluating IS (all features) on dataset_B...")
    model_IS_eval_B = train_eval_IS(Xtr, ytr, ptr, X_B, y_B, p_ids_B)
    voi_m_B = model_IS_eval_B[1]
    pat_m_B = model_IS_eval_B[2]
    print_evaluation_results("Instance-Space (All features) - B test", voi_m_B, pat_m_B)
    
    print(f"\nEvaluating IS (top {TOP_K_FEATURES} features) on dataset_B...")
    model_IS2_eval_B = train_eval_IS(
        Xtr[top5], ytr, ptr,
        X_B[top5], y_B, p_ids_B
    )
    voi_m2_B = model_IS2_eval_B[1]
    pat_m2_B = model_IS2_eval_B[2]
    print_evaluation_results(f"Instance-Space (Top {TOP_K_FEATURES} features) - B test", voi_m2_B, pat_m2_B)
    
    print("\nEvaluating ES (all features) on dataset_B...")
    Xpb, ypb = create_patient_embedding(X_B, y_B, p_ids_B)
    pat_m_es_B = train_eval_ES(Xptr, yptr, Xpb, ypb)[1]
    print_evaluation_results("Embedded-Space (All features) - B test", None, pat_m_es_B)
    
    print(f"\nEvaluating ES (top {TOP_K_FEATURES} features) on dataset_B...")
    Xpb_top5, ypb_top5 = create_patient_embedding(X_B[top5], y_B, p_ids_B)
    pat_m_es2_B = train_eval_ES(Xptr_top5, yptr_top5, Xpb_top5, ypb_top5)[1]
    print_evaluation_results(f"Embedded-Space (Top {TOP_K_FEATURES} features) - B test", None, pat_m_es2_B)
    
    # ============================================================================
    # PHASE 2: Cross-validation on combined dataset (A + B) WITH sex/age
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 2: CROSS-VALIDATION ON COMBINED DATASET (A+B) WITH SEX/AGE")
    print("="*80)
    
    print("\nCombining dataset_A (full) and dataset_B (adding NaN for sex/age)...")
    # Load full dataset_A with sex/age
    X_A_full, y_A_full, p_ids_A_full, feat_A_full = load_data(DATASET_A_PATH)
    
    # Add NaN columns to dataset_B for sex/age
    X_B_full = X_B.copy()
    if "sex" not in X_B_full.columns:
        X_B_full["sex"] = np.nan
    if "age" not in X_B_full.columns:
        X_B_full["age"] = np.nan
    
    # Reorder B to match A's column order
    X_B_full = X_B_full[X_A_full.columns]
    
    X_combined, y_combined, p_combined = combine_datasets(X_A_full, y_A_full, p_ids_A_full, X_B_full, y_B, p_ids_B)
    print_dataset_info(X_combined, y_combined, p_combined, feat_A_full)
    
    print_cv_header("Instance-Space (IS) on combined dataset (with sex/age)", N_SPLITS, RANDOM_STATE)
    cv_IS(X_combined, y_combined, p_combined, N_SPLITS)
    
    print_cv_header("Embedded-Space (ES) on combined dataset (with sex/age)", N_SPLITS, RANDOM_STATE)
    Xp_combined, yp_combined = create_patient_embedding(X_combined, y_combined, p_combined)
    cv_ES(Xp_combined, yp_combined, N_SPLITS)
    
    # ============================================================================
    # PHASE 3: Cross-validation on combined dataset (A + B) WITHOUT sex/age
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 3: CROSS-VALIDATION ON COMBINED DATASET (A+B) WITHOUT SEX/AGE")
    print("="*80)
    
    print("\nLoading dataset_A without sex/age...")
    X_A_no_demo, y_A_no_demo, p_ids_A_no_demo, feat_A_no_demo = load_data_exclude_features(
        DATASET_A_PATH, exclude_cols=EXCLUDE_FEATURES
    )
    print(f"Features after exclusion: {len(feat_A_no_demo)}")
    
    print("\nLoading dataset_B (already without sex/age)...")
    X_B_no_demo = X_B.copy()
    
    print("\nCombining dataset_A and dataset_B (without sex/age)...")
    X_combined_no_demo, y_combined_no_demo, p_combined_no_demo = combine_datasets(
        X_A_no_demo, y_A_no_demo, p_ids_A_no_demo,
        X_B_no_demo, y_B, p_ids_B
    )
    print_dataset_info(X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, feat_A_no_demo)
    
    print_cv_header("Instance-Space (IS) on combined dataset (without sex/age)", N_SPLITS, RANDOM_STATE)
    cv_IS(X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, N_SPLITS)
    
    print_cv_header("Embedded-Space (ES) on combined dataset (without sex/age)", N_SPLITS, RANDOM_STATE)
    Xp_combined_no_demo, yp_combined_no_demo = create_patient_embedding(
        X_combined_no_demo, y_combined_no_demo, p_combined_no_demo
    )
    cv_ES(Xp_combined_no_demo, yp_combined_no_demo, N_SPLITS)
    
    # ============================================================================
    # COMPILE RESULTS
    # ============================================================================
    results = {
        "dataset_A_internal": {
            "IS": {
                "voi_acc": voi_m[0],
                "voi_f1": voi_m[1],
                "pat_acc": pat_m[0],
                "pat_f1": pat_m[1],
            },
            "IS_top5": {
                "voi_acc": voi_m2[0],
                "voi_f1": voi_m2[1],
                "pat_acc": pat_m2[0],
                "pat_f1": pat_m2[1],
                "features": top5,
            },
            "ES": {
                "pat_acc": pat_m_es[0],
                "pat_f1": pat_m_es[1],
            },
            "ES_top5": {
                "pat_acc": pat_m_es2[0],
                "pat_f1": pat_m_es2[1],
                "features": top5,
            }
        },
        "dataset_B_external_validation": {
            "IS": {
                "voi_acc": voi_m_B[0],
                "voi_f1": voi_m_B[1],
                "pat_acc": pat_m_B[0],
                "pat_f1": pat_m_B[1],
            },
            "IS_top5": {
                "voi_acc": voi_m2_B[0],
                "voi_f1": voi_m2_B[1],
                "pat_acc": pat_m2_B[0],
                "pat_f1": pat_m2_B[1],
                "features": top5,
            },
            "ES": {
                "pat_acc": pat_m_es_B[0],
                "pat_f1": pat_m_es_B[1],
            },
            "ES_top5": {
                "pat_acc": pat_m_es2_B[0],
                "pat_f1": pat_m_es2_B[1],
                "features": top5,
            }
        }
    }
    
    print_final_summary(results["dataset_A_internal"])
    print("\n" + "="*80)
    print(" EXTERNAL VALIDATION ON DATASET_B")
    print("="*80)
    print_final_summary(results["dataset_B_external_validation"])
    
    plot_performance_comparison(results["dataset_A_internal"], f"{RESULTS_DIR}/performance_comparison_A.png")
    plot_performance_comparison(results["dataset_B_external_validation"], f"{RESULTS_DIR}/performance_comparison_B.png")
    
    save_report(results, f"{RESULTS_DIR}/summary_complete.json")
    print_completion_message()


if __name__ == "__main__":
    main()