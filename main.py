from data_utils import load_and_preprocess_data
from model_utils import (cross_validate_model, train_final_model, 
                         evaluate_model, explain_with_shap, get_top_shap_features)
from visualization_utils import plot_model_comparison, create_comparison_summary
import numpy as np
import os

def main():
    # Crea cartella results se non esiste
    os.makedirs("results", exist_ok=True)
    
    # ========================================
    # 1. TRAINING CON TUTTE LE FEATURES
    # ========================================
    X_train, X_test, y_train, y_test, groups_train, le = load_and_preprocess_data("dataset_A.csv")
    
    print("\n" + "="*50)
    print("TRAINING WITH ALL FEATURES (110)")
    print("="*50)
    cross_validate_model(X_train, y_train, groups_train, le)
    model_full = train_final_model(X_train, y_train, num_classes=4)
    evaluate_model(model_full, X_test, y_test, le)
    
    # Predizioni modello completo
    y_pred_full = model_full.predict(X_test)
    
    # SHAP analysis
    X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    shap_values_2d = explain_with_shap(model_full, X_sample, save_prefix="results/shap_all_features")
    
    # ========================================
    # 2. ESTRAI TOP 5 FEATURES
    # ========================================
    top_features = get_top_shap_features(shap_values_2d, X_sample.columns.tolist(), top_n=5)
    
    # ========================================
    # 3. RETRAIN CON TOP 5 FEATURES
    # ========================================
    print("\n" + "="*50)
    print("RETRAINING WITH TOP 5 SHAP FEATURES")
    print("="*50)
    
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]
    
    print(f"\nReduced feature set: {top_features}")
    print(f"Training shape: {X_train_reduced.shape}")
    print(f"Test shape: {X_test_reduced.shape}")
    
    cross_validate_model(X_train_reduced, y_train, groups_train, le)
    model_reduced = train_final_model(X_train_reduced, y_train, num_classes=4)
    evaluate_model(model_reduced, X_test_reduced, y_test, le)
    
    # Predizioni modello ridotto
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    
    # SHAP sul modello ridotto
    X_sample_reduced = X_test_reduced.sample(n=min(200, len(X_test_reduced)), random_state=42)
    explain_with_shap(model_reduced, X_sample_reduced, save_prefix="results/shap_top5_features")
    
    # ========================================
    # 4. VISUALIZZAZIONE COMPARATIVA
    # ========================================
    print("\n" + "="*50)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*50)
    
    # Grafico completo di confronto
    plot_model_comparison(
        y_test=y_test,
        y_pred_full=y_pred_full,
        y_pred_reduced=y_pred_reduced,
        classes=le.classes_,
        save_path="results/model_comparison.png"
    )
    
    # Report testuale
    create_comparison_summary(
        y_test=y_test,
        y_pred_full=y_pred_full,
        y_pred_reduced=y_pred_reduced,
        classes=le.classes_,
        save_path="results/comparison_summary.txt"
    )
    
    print("\n" + "="*50)
    print(" ANALYSIS COMPLETE!")
    print("="*50)
    print("\nGenerated files in results/:")
    print("   model_comparison.png - Complete visual comparison")
    print("   comparison_summary.txt - Detailed text report")
    print("   shap_all_features_*.png - SHAP plots for full model")
    print("   shap_top5_features_*.png - SHAP plots for reduced model")
    print("\n" + "="*50)
    
    # Quick summary
    acc_full = (y_test == y_pred_full).mean()
    acc_reduced = (y_test == y_pred_reduced).mean()
    
    print(f"\n  QUICK SUMMARY:")
    print(f"   Full Model (110 features):    Accuracy = {acc_full:.3f}")
    print(f"   Reduced Model (5 features):   Accuracy = {acc_reduced:.3f}")
    print(f"   Difference:                   {acc_reduced - acc_full:+.3f}")
    print(f"\n   Top 5 Features: {', '.join(top_features)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()