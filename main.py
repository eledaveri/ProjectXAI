from data_utils import load_and_preprocess_data
from model_utils import (cross_validate_model, train_final_model, 
                         evaluate_model, explain_with_shap, get_top_shap_features)
import numpy as np

def main():
    # Training completo
    X_train, X_test, y_train, y_test, groups_train, le = load_and_preprocess_data("dataset_A.csv")
    
    print("\n" + "="*50)
    print("TRAINING WITH ALL FEATURES")
    print("="*50)
    cross_validate_model(X_train, y_train, groups_train, le)
    model = train_final_model(X_train, y_train, num_classes=4)
    evaluate_model(model, X_test, y_test, le)
    
    # SHAP analysis
    X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    shap_values_2d = explain_with_shap(model, X_sample, save_prefix="results/shap_all_features")
    
    # Estrai top 5 features
    top_features = get_top_shap_features(shap_values_2d, X_sample.columns.tolist(), top_n=5)
    
    # Retrain con solo le top 5 features
    print("\n" + "="*50)
    print("RETRAINING WITH TOP 5 SHAP FEATURES")
    print("="*50)
    
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]
    
    print(f"\nReduced feature set: {top_features}")
    print(f"Shape: {X_train_reduced.shape}")
    
    cross_validate_model(X_train_reduced, y_train, groups_train, le)
    model_reduced = train_final_model(X_train_reduced, y_train, num_classes=4)
    evaluate_model(model_reduced, X_test_reduced, y_test, le)
    
    # SHAP sul modello ridotto
    X_sample_reduced = X_test_reduced.sample(n=min(200, len(X_test_reduced)), random_state=42)
    explain_with_shap(model_reduced, X_sample_reduced, save_prefix="results/shap_top5_features")


if __name__ == "__main__":
    main()
