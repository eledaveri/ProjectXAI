from data_utils import load_data, split_by_patient
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
from config import TEST_SIZE, N_SPLITS, RANDOM_STATE


def main():

    # Load data
    print("\nLoading dataset...")
    X, y, p_ids, feat = load_data("dataset_A.csv")
    print_dataset_info(X, y, p_ids, feat)
    
    # Split data by patient (stratified)
    print("\nSplitting data by patient (stratified)...")
    Xtr, ytr, ptr, Xte, yte, pte = split_by_patient(
        X, y, p_ids, TEST_SIZE, RANDOM_STATE
    )
    print_split_info(Xtr, ytr, ptr, Xte, yte, pte)
    
    # Cross-validation: Instance-Space
    print_cv_header("Instance-Space (IS)", N_SPLITS, RANDOM_STATE)
    cv_IS(X, y, p_ids, N_SPLITS)
    
    # Cross-validation: Embedded-Space
    print_cv_header("Embedded-Space (ES)", N_SPLITS, RANDOM_STATE)
    Xp, yp = create_patient_embedding(X, y, p_ids)
    cv_ES(Xp, yp, N_SPLITS)
    
    # Train and evaluate Instance-Space model
    print("\nTraining Instance-Space model on train set...")
    model_IS, voi_m, pat_m = train_eval_IS(Xtr, ytr, ptr, Xte, yte, pte)
    print_evaluation_results("Instance-Space (All features)", voi_m, pat_m)
    
    # SHAP analysis for feature importance
    print("\nComputing SHAP values for feature importance...")
    shap_vals = explain_shap(model_IS, Xtr, "results/IS")
    top5 = top_shap_features(shap_vals, feat, 5)
    print_shap_results(top5, k=5)
    
    # Retrain Instance-Space with top 5 features
    print("\nRetraining Instance-Space model with top 5 features...")
    model_IS2, voi_m2, pat_m2 = train_eval_IS(
        Xtr[top5], ytr, ptr,
        Xte[top5], yte, pte
    )
    print_evaluation_results("Instance-Space (Top 5 features)", voi_m2, pat_m2)
    
    # Train Embedded-Space with ALL features
    print("\nTraining Embedded-Space model with all features...")
    Xptr, yptr = create_patient_embedding(Xtr, ytr, ptr)
    Xpte, ypte = create_patient_embedding(Xte, yte, pte)
    model_ES, pat_m_es = train_eval_ES(Xptr, yptr, Xpte, ypte)
    print_evaluation_results("Embedded-Space (All features)", None, pat_m_es)
    
    # Train Embedded-Space with top 5 features only
    print("\nTraining Embedded-Space model with top 5 features...")
    Xptr_top5, yptr_top5 = create_patient_embedding(Xtr[top5], ytr, ptr)
    Xpte_top5, ypte_top5 = create_patient_embedding(Xte[top5], yte, pte)
    model_ES2, pat_m_es2 = train_eval_ES(Xptr_top5, yptr_top5, Xpte_top5, ypte_top5)
    print_evaluation_results("Embedded-Space (Top 5 features)", None, pat_m_es2)
    
    # Compile all results
    results = {
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
}

    
    # Print final summary and save results
    print_final_summary(results)
    plot_performance_comparison(results, "results/performance_comparison.png")
    save_report(results, "results/summary.json")
    print_completion_message()


if __name__ == "__main__":
    main()