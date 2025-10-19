from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt


def create_xgb_model(num_classes=4):
    return XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )

def cross_validate_model(X_train, y_train, groups_train, label_encoder):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(
        sgkf.split(X_train, y_train, groups=groups_train)
    ):
        print(f"\nFold {fold + 1}")
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model = create_xgb_model(num_classes=len(label_encoder.classes_))
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        print(classification_report(
            y_val,
            preds,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0
        ))

def train_final_model(X_train, y_train, num_classes):
    model = create_xgb_model(num_classes)
    model.fit(X_train, y_train)
    print(f"✅ Modello finale addestrato con {num_classes} classi.")
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    print("\nRisultati sul test set:")
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

def explain_with_shap(model, X_sample, save_prefix="results/shap"):
    """Calcola e salva grafici SHAP come PNG (compatibile con SHAP 0.49 e multi-class)"""

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Se l'oggetto non è un Explanation, lo convertiamo
    if not isinstance(shap_values, shap.Explanation):
        shap_values = shap.Explanation(values=shap_values, data=X_sample)

    # --- Gestione multi-class ---
    if shap_values.values.ndim == 3:
        # (n_sample, n_class, n_feature)
        n_classes = shap_values.values.shape[1]
        print(f"🔹 Rilevato modello multi-classe con {n_classes} classi.")
        # scegliamo la classe 0 (puoi modificarla)
        class_idx = 0  
        shap_values_class = shap.Explanation(
            values=shap_values.values[:, class_idx, :],
            base_values=shap_values.base_values[:, class_idx] if shap_values.base_values.ndim > 1 else shap_values.base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    else:
        shap_values_class = shap_values

    # --- Summary Plot ---
    shap.summary_plot(shap_values_class, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Bar Plot ---
    shap.summary_plot(shap_values_class.values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Waterfall Plot (per il primo campione e la prima classe) ---
    shap.plots.waterfall(shap_values_class[0], show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Grafici SHAP salvati in: {os.path.dirname(save_prefix)}/")

    return shap_values_class
