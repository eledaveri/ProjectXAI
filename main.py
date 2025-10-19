from data_utils import load_and_preprocess_data
from model_utils import cross_validate_model, train_final_model, evaluate_model, explain_with_shap
import numpy as np

def main():
    X_train, X_test, y_train, y_test, groups_train, le = load_and_preprocess_data("dataset_A.csv")

    cross_validate_model(X_train, y_train, groups_train, le)
    model = train_final_model(X_train, y_train, num_classes=4)
    evaluate_model(model, X_test, y_test, le)

    # Usa un piccolo campione per SHAP
    X_sample = X_test.sample(n=min(200, len(X_test)),random_state=42)
    explain_with_shap(model, X_sample, save_prefix="results/shap")

if __name__ == "__main__":
    main()
