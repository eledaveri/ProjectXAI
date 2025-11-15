# ProjectXAI: Hodgkin Lymphoma Classification with XAI

A comprehensive machine learning pipeline for lymphoma classification (Hodgkin Lymphoma vs Others) using two distinct approaches: Instance-Space (IS) and Embedded-Space (ES), with SHAP-based explainability. Includes multi-phase validation strategy with internal and external cohorts.

## Project Structure

```
project/
│
├── config.py                     # Configuration parameters (paths, CV folds, random state)
├── data_utils.py                 # Data loading, patient-stratified splitting
├── model_IS.py                   # Instance-Space model (VOI-level predictions → patient aggregation)
├── model_ES.py                   # Embedded-Space model (patient-level embeddings)
├── shap_utils.py                 # SHAP explainability utilities
├── evaluation.py                 # Metrics computation and reporting
├── visualization_utils.py        # Formatted output and plotting functions
├── main.py                       # Main experimental pipeline (3 phases)
│
├── data/                         # Input datasets
│   ├── dataset_A.csv             # Internal dataset (36 patients)
│   ├── dataset_B.csv             # External validation dataset
│   └── dataset_A+B.csv           # Combined dataset (A+B, for Phase 2-3)
│
└── results/                      # Output directory (auto-generated)
    ├── summary_complete.json              # Complete results (all phases)
    ├── performance_comparison_A.png       # Model comparison (internal A test)
    ├── performance_comparison_B.png       # Model comparison (external B test)
    ├── IS_summary.png                    # SHAP feature impact
    └── IS_bar.png                        # SHAP feature ranking
```

## Project Overview

### Datasets

#### Dataset A (Internal)
- **Path**: `data/dataset_A.csv`
- **36 patients** with **349 VOIs (Volumes of Interest)**
- **111 radiomic features** per VOI
- **Binary classification**: HL (Hodgkin Lymphoma) vs Others
- **Class imbalance**: 
  - VOI-level: 15.2% HL, 84.8% Others
  - Patient-level: 25.0% HL, 75.0% Others
- **Variable VOIs per patient**: 1-29 (mean: 9.69, median: 8.0)
- **Usage**: Training and internal validation (70/30 split)

#### Dataset B (External Validation)
- **Path**: `data/dataset_B.csv`
- Independent external cohort
- Same feature set and classification task
- **Usage**: External validation of models trained on Dataset A (Phase 1)

#### Dataset A+B (Combined)
- **Path**: `data/dataset_A+B.csv`
- Preexisting combined file (Dataset A + Dataset B merged)
- **Usage**: Cross-validation on larger combined dataset (Phase 2-3)
- Same features as individual datasets

### Experimental Design

Three-phase validation strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 1: INTERNAL & EXTERNAL                   │
├─────────────────────────────────────────────────────────────────┤
│ Files: dataset_A.csv + dataset_B.csv                            │
│                                                                  │
│ Train on Dataset A (70%)                                        │
│ ├─ Test on Dataset A (30% internal test)                        │
│ └─ Test on Dataset B (external validation)                      │
│                                                                  │
│ [4 model configurations per test:                               │
│  IS/all, IS/top5, ES/all, ES/top5]                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│     PHASE 2: CV ON COMBINED DATASET WITH DEMOGRAPHICS            │
├─────────────────────────────────────────────────────────────────┤
│ File: dataset_A+B.csv                                            │
│                                                                  │
│ Cross-validation on A+B (3-fold, stratified)                   │
│ Keep all features (including sex/age)                           │
│ Evaluate IS and ES approaches                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│    PHASE 3: CV ON COMBINED DATASET WITHOUT DEMOGRAPHICS          │
├─────────────────────────────────────────────────────────────────┤
│ File: dataset_A+B.csv                                            │
│                                                                  │
│ Cross-validation on A+B (3-fold, stratified)                   │
│ Exclude sex/age features                                        │
│ Evaluate impact of demographic features                         │
└─────────────────────────────────────────────────────────────────┘
```

## Methodology

### Two Complementary Approaches

#### 1. **Instance-Space (IS)**
- Operates at VOI level (individual volume of interest)
- XGBoost classifier predicts each VOI independently
- Aggregates VOI predictions to patient level via majority voting
- Suitable for fine-grained, VOI-level analysis
- Provides interpretability at multiple levels

#### 2. **Embedded-Space (ES)**
- Aggregates VOI features per patient (min, max, mean, std)
- Creates patient-level embeddings (dimensionality: 111 × 4 = 444)
- Single XGBoost prediction per patient
- Reduces noise and variability across VOIs
- More efficient for patient-level decisions

### Model Configuration

Both approaches use XGBoost with identical hyperparameters:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Cross-Validation Strategy

- **3-fold stratified cross-validation** on patient level
- Uses `StratifiedGroupKFold` to respect patient-level stratification
- Prevents patient leakage between training and validation folds
- Maintains class distribution in each fold

### Feature Selection

- **SHAP analysis** identifies top-K most important features
- Default K=5 features from 111 available
- Compares full model vs. reduced model performance
- Assesses feature redundancy

## Configuration

Edit `config.py` to customize:

```python
# Data split
TEST_SIZE = 0.3              # Train/test split ratio for Phase 1
N_SPLITS = 3                 # Number of CV folds (Phases 2 & 3)
RANDOM_STATE = 42            # Reproducibility seed

# Dataset paths
DATASET_A_PATH = "data/dataset_A.csv"         # Internal training dataset
DATASET_B_PATH = "data/dataset_B.csv"         # External validation dataset
DATASET_AB_PATH = "data/dataset_A+B.csv"      # Combined dataset for final CV

# Feature engineering
EXCLUDE_FEATURES = ["sex", "age"]  # Features to exclude in Phase 3
TOP_K_FEATURES = 5                 # Features to select in Phase 1

# Output
RESULTS_DIR = "results"
SHAP_PREFIX = "IS"
```

## Usage

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib joblib
```

### Dataset Setup

Ensure the `data/` directory contains all three files:

```
data/
├── dataset_A.csv           # Internal (required for Phase 1)
├── dataset_B.csv           # External validation (required for Phase 1)
└── dataset_A+B.csv         # Combined (required for Phases 2-3)
```

### Run Full Pipeline (All 3 Phases)

```bash
python main.py
```

This executes sequentially:
1. **Phase 1**: Train on A (70%), test on A (30%), validate on B
2. **Phase 2**: Cross-validation on A+B with all features (including demographics)
3. **Phase 3**: Cross-validation on A+B without demographic features (sex/age)

### Output Files

All results are saved to `results/` directory:

```
results/
├── summary_complete.json              # Consolidated results
├── performance_comparison_A.png       # Models on internal A test set
├── performance_comparison_B.png       # Models on external B test set
├── IS_summary.png                    # SHAP feature impact
└── IS_bar.png                        # SHAP feature ranking
```

## Results

### Phase 1: Internal vs. External Validation

#### Dataset A (Internal Test)
Models trained and tested on Dataset A (70/30 split):

| Approach | Features | VOI Acc | VOI F1 | Patient Acc | Patient F1 |
|----------|----------|---------|--------|-------------|------------|
| **IS** | All (111) | 0.9111 | 0.5000 | 0.8182 | 0.5000 |
| **IS** | Top 5 | 0.9000 | 0.4000 | 0.7273 | 0.0000 |
| **ES** | All (111) | - | - | 1.0000 | 1.0000 |
| **ES** | Top 5 | - | - | 1.0000 | 1.0000 |

#### Dataset B (External Validation)
Models trained on Dataset A, tested on external Dataset B:

| Approach | Features | VOI Acc | VOI F1 | Patient Acc | Patient F1 |
|----------|----------|---------|--------|-------------|------------|
| **IS** | All (111) | 0.1154 | 0.2069 | 0.0417 | 0.0800 |
| **IS** | Top 5 | 0.2179 | 0.3579 | 0.1250 | 0.2222 |
| **ES** | All (111) | - | - | 1.0000 | 1.0000 |
| **ES** | Top 5 | - | - | 1.0000 | 1.0000 |

#### Cross-Validation (3-fold) on Dataset A

**Instance-Space (IS)**
```
Patient-level metrics (aggregated from VOI predictions):
  Accuracy: 0.833 ± 0.068
  F1-score: 0.667 ± 0.000
```

**Embedded-Space (ES)**
```
Patient-level metrics:
  Accuracy: 1.000 ± 0.000
  F1-score: 1.000 ± 0.000
```

### Phase 2: Combined Dataset WITH Demographics (sex/age)

Cross-validation on dataset_A+B.csv with all demographic features:

- **IS model**: Patient-level F1 = [reported in output]
- **ES model**: Patient-level F1 = [reported in output]
- **Interpretation**: Baseline performance with full feature set

### Phase 3: Combined Dataset WITHOUT Demographics

Cross-validation on dataset_A+B.csv excluding sex/age:

- **IS model**: Patient-level F1 = [reported in output]
- **ES model**: Patient-level F1 = [reported in output]
- **Interpretation**: Impact of demographic feature removal

## 🔍 SHAP Explainability

### Top 5 Most Important Features

From SHAP analysis on Dataset A training set:

1. **age** - Patient age (demographic)
2. **sex** - Patient sex 0=M, 1=F (demographic)
3. **A41** - Radiomic feature
4. **A63** - Radiomic feature
5. **A92** - Radiomic feature

### SHAP Visualizations

Generated files (auto-saved):

- **results/IS_bar.png** - Feature importance ranking
- **results/IS_summary.png** - Feature impact on predictions
- **results/performance_comparison_A.png** - Internal test comparison
- **results/performance_comparison_B.png** - External test comparison

### Key Insights

- ✅ Using **top 5 features** maintains comparable performance to full feature set
- ⚠️ **IS model generalizes poorly** to external dataset
- ✅ **ES model maintains consistent performance** on both internal and external
- 📊 **Demographic features important**: Age and sex among top predictors
- 🔗 **High feature redundancy**: 111 features can be reduced to 5 with similar performance

## ⚠️ Limitations & Warnings

1. **Small Dataset**: Only 36 patients in Dataset A
   - High feature-to-sample ratio (111:36 ≈ 3:1)
   - Increased overfitting risk

2. **Perfect ES Accuracy**: F1=1.0 on both internal and external tests
   - Unlikely in real-world scenario
   - Possible indicators: data leakage, small validation set

3. **Poor IS Generalization**: 81.8% → 4.2% accuracy drop
   - Suggests Dataset A/B mismatch
   - May indicate preprocessing differences

4. **Class Imbalance**: Only 25% positive class
   - May skew metrics
   - F1-score preferred over accuracy

5. **External Validation Required**: 
   - Results should be validated on larger independent cohort
   - Current findings may not generalize broadly

## Implementation Details

### Patient-Level Aggregation (Instance-Space)

**Majority Voting Strategy** (default):
```python
def aggregate_majority(y_pred, patient_ids):
    return df.groupby("patient")["pred"].agg(
        lambda x: x.value_counts().idxmax()
    )
```

### Patient-Level Embedding (Embedded-Space)

Statistical aggregation per patient:
```python
def create_patient_embedding(X, y, patient_ids):
    agg = df.groupby("patient").agg(["min", "max", "mean", "std"])
    # Output: (n_patients, 111 × 4) = (n_patients, 444)
    return agg, y_patient
```

### Data Alignment

When combining datasets with different features:
```python
def align_features(X_source, X_target):
    # Keep only common features, in source order
    common = [col for col in X_source.columns if col in X_target.columns]
    return X_target[common]
```

## Model Saving & Loading

```python
from joblib import dump, load
from model_IS import save_model

# Save
save_model(model, "results/model_IS.joblib")

# Load
model = load("results/model_IS.joblib")
predictions = model.predict(X_new)
```

## SHAP Analysis

```python
from shap_utils import explain_shap, top_shap_features

# Compute SHAP values
shap_vals = explain_shap(model, X_train, "results/IS")

# Get top features
top5 = top_shap_features(shap_vals, feature_names, k=5)

# Visualizations auto-saved
# - results/IS_summary.png
# - results/IS_bar.png
```



