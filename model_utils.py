from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import os
import numpy as np  

def create_xgb_model(num_classes=4):
    """Create an XGBClassifier with specified parameters.
    Args:
        num_classes: Number of unique classes in the target variable.
    Returns:
        Configured XGBClassifier instance.
    """
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
    """Execute Stratified Group K-Fold Cross-Validation and print classification reports.
    Args:
        X_train: Training features.
        y_train: Training labels.
        groups_train: Group labels for the training set.
        label_encoder: Fitted LabelEncoder for decoding class labels.    
    """
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
    """Train the final model on the entire training set.
    Args:
        X_train: Training features.
        y_train: Training labels.
        num_classes: Number of unique classes in the target variable.
    Returns:
        Trained XGBClassifier model.
    """
    model = create_xgb_model(num_classes)
    model.fit(X_train, y_train)
    print(f"Final model trained with {num_classes} classes.")
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    print("\nResults on test set:")
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

def explain_with_shap(model, X_sample, save_prefix="results/shap"):
    """Compute SHAP values and generate explanation plots.
    Args:
        model: Trained model to explain.
        X_sample: Sample of input data for SHAP explanations.
        save_prefix: Prefix for saving SHAP plots.
    Returns:
        SHAP values array.
    """
    # TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    print(f"Debug shape:")
    print(f"shap_values type: {type(shap_values)}")
    print(f"shap_values.values shape: {shap_values.values.shape}")
    print(f"X_sample shape: {X_sample.shape}")
    
    # --- Multi-class---
    # XGBoost multi-class, shap_values.values has shape (n_samples, n_features, n_classes)
    # or (n_samples, n_classes, n_features) depending on version.
    
    if shap_values.values.ndim == 3:
        # Determine the shape format
        n_features = X_sample.shape[1]
        shape = shap_values.values.shape
        
        # Check which dimension corresponds to features
        if shape[1] == n_features:
            # Format: (n_samples, n_features, n_classes)
            print(f"Format: (n_samples={shape[0]}, n_features={shape[1]}, n_classes={shape[2]})")
            # Transpose to (n_samples, n_classes, n_features) if needed
            values_transposed = shap_values.values
            n_classes = shape[2]
        elif shape[2] == n_features:
            # Format: (n_samples, n_classes, n_features)
            print(f"Format: (n_samples={shape[0]}, n_classes={shape[1]}, n_features={shape[2]})")
            values_transposed = shap_values.values
            n_classes = shape[1]
        else:
            raise ValueError(f"Cannot match n_features={n_features} with shape {shape}")
        
        print(f"Detect a multi-class model with {n_classes} classes.")
        
        # Extract SHAP values for the first class as example
        class_idx = 0
        if shape[1] == n_features:
            # (n_samples, n_features, n_classes) -> (n_samples, n_features)
            shap_values_2d = shap_values.values[:, :, class_idx]
        else:
            # (n_samples, n_classes, n_features) -> (n_samples, n_features)
            shap_values_2d = shap_values.values[:, class_idx, :]
        
        print(f" Extracted valeus for the class {class_idx}, shape: {shap_values_2d.shape}")
        
    else:
        # Binary or already 2D
        shap_values_2d = shap_values.values
        print(f"Binry/2D model, shape: {shap_values_2d.shape}")
    
    # --- Summary Plot (Beeswarm) ---
    plt.figure()
    shap.summary_plot(shap_values_2d, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}_summary.png")
    
    # --- Bar Plot ---
    plt.figure()
    shap.summary_plot(shap_values_2d, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}_bar.png")
    
    # --- Waterfall Plot  ---
    plt.figure()
    # Extract base value for the selected class
    if shap_values.base_values.ndim > 1:
        # Multi-class: shape (n_samples, n_classes)
        base_value = shap_values.base_values[0, class_idx]
    elif len(shap_values.base_values) > 1:
        # Array of base values
        base_value = shap_values.base_values[0]
    else:
        # Single base value
        base_value = float(shap_values.base_values)
    
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_2d[0],
            base_values=base_value,
            data=X_sample.iloc[0].values,
            feature_names=X_sample.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_prefix}_waterfall.png")
    
    print(f"\nAll shap graphs saved in: {os.path.dirname(save_prefix)}/")
    
    return shap_values_2d

def get_top_shap_features(shap_values_2d, feature_names, top_n=5):
    """Extract the top N most important features based on mean absolute SHAP values.
    
    Args:
        shap_values_2d: 2D array of SHAP values (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        List of top feature names
    """
    mean_abs_shap = np.abs(shap_values_2d).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    print(f"\nTop {top_n} features by SHAP importance:")
    for i, (feat, importance) in enumerate(zip(top_features, mean_abs_shap[top_indices])):
        print(f"  {i+1}. {feat}: {importance:.4f}")
    
    return top_features