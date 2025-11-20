from data_utils import (
    load_data, load_data_exclude_features, split_by_patient, 
    align_features, merge_datasets_with_demographics
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
    plot_performance_comparison,
    plot_cv_comparison,
    plot_demographics_impact,
    plot_all_phases_comparison,
    plot_feature_distribution,
    plot_feature_scatter
)
from config import (
    TEST_SIZE, N_SPLITS, RANDOM_STATE,
    DATASET_A_PATH, DATASET_B_PATH, DATASET_AB_PATH,
    EXCLUDE_FEATURES, TOP_K_FEATURES,
    RESULTS_DIR, SHAP_PREFIX, MERGE_STRATEGY
)
import pandas as pd
import numpy as np


def main():

    # ============================================================================
    # PHASE 1: Train on dataset_A, validate on dataset_B (external)
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 1: TRAIN ON DATASET_A, VALIDATE ON DATASET_B")
    print("="*80)
    
    # Load data
    print("\nLoading dataset_A (internal)...")
    X_A, y_A, p_ids_A, feat_A = load_data(DATASET_A_PATH)
    print_dataset_info(X_A, y_A, p_ids_A, feat_A)
    
    print("\nLoading dataset_B (external validation)...")
    X_B, y_B, p_ids_B, feat_B = load_data(DATASET_B_PATH)
    print_dataset_info(X_B, y_B, p_ids_B, feat_B)
    
    # Align dataset_A to dataset_B (remove features not in B)
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
    # VALIDATION ON DATASET_B (EXTERNAL)
    # ============================================================================
    print("\n" + "="*80)
    print(" VALIDATION ON DATASET_B (EXTERNAL)")
    print(" (models trained on dataset_A)")
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
    # PHASE 2: Cross-validation on dataset_A+B WITH demographics (sex/age)
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 2: CROSS-VALIDATION ON DATASET_A+B WITH SEX/AGE")
    print(f" (Merge strategy: {MERGE_STRATEGY})")
    print("="*80)
    
    # Merge datasets using configured strategy
    X_combined_with_demo, y_combined_with_demo, p_combined_with_demo, feat_combined_with_demo = \
        merge_datasets_with_demographics(DATASET_A_PATH, DATASET_B_PATH, strategy=MERGE_STRATEGY)
    print_dataset_info(X_combined_with_demo, y_combined_with_demo, p_combined_with_demo, feat_combined_with_demo)
    
    print_cv_header("Instance-Space (IS) on dataset_A+B (with demographics)", N_SPLITS, RANDOM_STATE)
    cv_is_with_demo = cv_IS(X_combined_with_demo, y_combined_with_demo, p_combined_with_demo, N_SPLITS)
    
    print_cv_header("Embedded-Space (ES) on dataset_A+B (with demographics)", N_SPLITS, RANDOM_STATE)
    Xp_combined_with_demo, yp_combined_with_demo = create_patient_embedding(
        X_combined_with_demo, y_combined_with_demo, p_combined_with_demo
    )
    cv_es_with_demo = cv_ES(Xp_combined_with_demo, yp_combined_with_demo, N_SPLITS)
    
    # ============================================================================
    # PHASE 3: Cross-validation on dataset_A+B WITHOUT demographics (sex/age)
    # ============================================================================
    print("\n" + "="*80)
    print(" PHASE 3: CROSS-VALIDATION ON DATASET_A+B WITHOUT SEX/AGE")
    print(" (Using dataset_A+B.csv file)")
    print("="*80)
    
    print("\nLoading dataset_A+B.csv (without sex/age)...")
    X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, feat_combined_no_demo = load_data(DATASET_AB_PATH)
    print(f"Features in dataset_A+B.csv: {len(feat_combined_no_demo)}")
    print_dataset_info(X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, feat_combined_no_demo)
    
    # Check if sex/age are present
    if 'sex' in feat_combined_no_demo or 'age' in feat_combined_no_demo:
        print("\nWARNING: sex/age found in dataset_A+B.csv, removing them...")
        X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, feat_combined_no_demo = \
            load_data_exclude_features(DATASET_AB_PATH, exclude_cols=EXCLUDE_FEATURES)
        print(f"Features after exclusion: {len(feat_combined_no_demo)}")
    
    print_cv_header("Instance-Space (IS) on dataset_A+B (without demographics)", N_SPLITS, RANDOM_STATE)
    cv_is_no_demo = cv_IS(X_combined_no_demo, y_combined_no_demo, p_combined_no_demo, N_SPLITS)
    
    print_cv_header("Embedded-Space (ES) on dataset_A+B (without demographics)", N_SPLITS, RANDOM_STATE)
    Xp_combined_no_demo, yp_combined_no_demo = create_patient_embedding(
        X_combined_no_demo, y_combined_no_demo, p_combined_no_demo
    )
    cv_es_no_demo = cv_ES(Xp_combined_no_demo, yp_combined_no_demo, N_SPLITS)
    
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
        },
        "dataset_AB_with_demographics": {
            "IS_cv": {
                "pat_acc": cv_is_with_demo["acc"].mean(),
                "pat_f1": cv_is_with_demo["f1"].mean(),
                "pat_acc_std": cv_is_with_demo["acc"].std(),
                "pat_f1_std": cv_is_with_demo["f1"].std(),
            },
            "ES_cv": {
                "pat_acc": cv_es_with_demo["acc"].mean(),
                "pat_f1": cv_es_with_demo["f1"].mean(),
                "pat_acc_std": cv_es_with_demo["acc"].std(),
                "pat_f1_std": cv_es_with_demo["f1"].std(),
            }
        },
        "dataset_AB_without_demographics": {
            "IS_cv": {
                "pat_acc": cv_is_no_demo["acc"].mean(),
                "pat_f1": cv_is_no_demo["f1"].mean(),
                "pat_acc_std": cv_is_no_demo["acc"].std(),
                "pat_f1_std": cv_is_no_demo["f1"].std(),
            },
            "ES_cv": {
                "pat_acc": cv_es_no_demo["acc"].mean(),
                "pat_f1": cv_es_no_demo["f1"].mean(),
                "pat_acc_std": cv_es_no_demo["acc"].std(),
                "pat_f1_std": cv_es_no_demo["f1"].std(),
            }
        }
    }
    
    print_final_summary(results["dataset_A_internal"])
    print("\n" + "="*80)
    print(" EXTERNAL VALIDATION ON DATASET_B")
    print("="*80)
    print_final_summary(results["dataset_B_external_validation"])
    
    # ============================================================================
    # GENERATE ALL VISUALIZATIONS
    # ============================================================================
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Original plots (Phase 1)
    plot_performance_comparison(results["dataset_A_internal"], f"{RESULTS_DIR}/performance_comparison_A.png")
    plot_performance_comparison(results["dataset_B_external_validation"], f"{RESULTS_DIR}/performance_comparison_B.png")
    
    # New plots (Phase 2 and 3 comparison)
    plot_cv_comparison(
        results["dataset_AB_with_demographics"],
        results["dataset_AB_without_demographics"],
        f"{RESULTS_DIR}/cv_demographics_comparison.png"
    )
    
    plot_demographics_impact(
        results["dataset_AB_with_demographics"],
        results["dataset_AB_without_demographics"],
        f"{RESULTS_DIR}/demographics_impact.png"
    )
    
    plot_all_phases_comparison(
        results,
        f"{RESULTS_DIR}/all_phases_comparison.png"
    )
    
    # ============================================================================
    # PHASE 4: Investigate overfitting (A41 vs A49)
    # ============================================================================
    
    print("\n" + "="*80)
    print(" INVESTIGATING ES OVERFITTING (A41 vs A49)")
    print("="*80)
    
    # Create patient-level embeddings for dataset_A
    Xp_A_all, yp_A_all = create_patient_embedding(X_A_aligned, y_A, p_ids_A)
    
    # Select features A41 and A49 at various aggregation levels
    features_to_plot = ["A41", "A49"] 
    
    # Plot of feature distributions at VOI level
    plot_feature_distribution(
        X_A_aligned, 
        pd.Series(y_A, index=X_A_aligned.index), 
        features_to_plot, 
        f"{RESULTS_DIR}/voi_distribution_A41_A49.png"
    )

    # Scatter plot of feature means at Patient level
    scatter_features = [f"{features_to_plot[0]}_mean", f"{features_to_plot[1]}_mean"]
    Xp_filtered = Xp_A_all[[f for f in scatter_features if f in Xp_A_all.columns]].copy()
    
    if len(Xp_filtered.columns) == 2:
        plot_feature_scatter(
            Xp_filtered, 
            yp_A_all, 
            scatter_features, 
            f"{RESULTS_DIR}/pat_scatter_A41_mean_A49_mean.png"
        )
    else:
        print(f"Warning: Could not find required aggregated features for scatter plot: {scatter_features}. Skipping.")


    # Save results
    save_report(results, f"{RESULTS_DIR}/summary_complete.json")
    
    # Print summary of generated files
    print("\n" + "="*80)
    print(" GENERATED FILES")
    print("="*80)
    print("\n Visualizations (PNG):")
    print("  1. performance_comparison_A.png - Dataset A internal test results")
    print("  2. performance_comparison_B.png - Dataset B external validation results")
    print("  3. cv_demographics_comparison.png - CV with/without demographics (F1 & Accuracy)")
    print("  4. demographics_impact.png - Impact of sex/age on performance (% change)")
    print("  5. all_phases_comparison.png - Comprehensive overview of all phases")
    print("  6. voi_distribution_A41_A49.png - Distribution of A41 and A49 (VOI level)") # Nuovo
    print("  7. pat_scatter_A41_mean_A49_mean.png - Scatter plot of mean(A41) vs mean(A49) (Patient level)") # Nuovo
    print("\n SHAP Analysis:")
    print("  8. IS_summary.png - SHAP feature importance summary")
    print("  9. IS_bar.png - SHAP feature importance bar chart")
    print("\n Data Files:")
    print("  10. summary_complete.json - Complete numerical results")
    
    print_completion_message()
    


if __name__ == "__main__":
    main()