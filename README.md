# ProjectXAI

project/
│
├── data_utils.py          # Data loading and pre-processing
├── model_utils.py         # Training function, cross-validation and SHAP
├── main.py                # Main script
└── dataset_A.csv

## MODEL PERFORMAMCES

```
Fold 1
              precision    recall  f1-score   support

       DLBCL       0.50      0.87      0.63        15
          FL       0.00      0.00      0.00         0
          HL       1.00      1.00      1.00        13
         MCL       1.00      0.56      0.72        32

    accuracy                           0.73        60
   macro avg       0.62      0.61      0.59        60
weighted avg       0.88      0.73      0.76        60


Fold 2
              precision    recall  f1-score   support

       DLBCL       0.90      0.75      0.82        12
          FL       0.26      1.00      0.41         6
          HL       1.00      1.00      1.00         7
         MCL       1.00      0.63      0.78        41

    accuracy                           0.73        66
   macro avg       0.79      0.85      0.75        66
weighted avg       0.91      0.73      0.77        66


Fold 3
              precision    recall  f1-score   support

       DLBCL       1.00      0.24      0.38        17
          FL       0.08      0.10      0.09        21
          HL       1.00      1.00      1.00         1
         MCL       0.16      0.21      0.18        29

    accuracy                           0.19        68
   macro avg       0.56      0.38      0.41        68
weighted avg       0.36      0.19      0.21        68


Fold 4
              precision    recall  f1-score   support

       DLBCL       0.00      0.00      0.00         1
          FL       0.00      0.00      0.00         0
          HL       1.00      0.43      0.60         7
         MCL       0.00      0.00      0.00         6

    accuracy                           0.21        14
   macro avg       0.25      0.11      0.15        14
weighted avg       0.50      0.21      0.30        14


Fold 5
              precision    recall  f1-score   support

       DLBCL       0.20      1.00      0.33         3
          FL       0.00      0.00      0.00        24
          HL       1.00      1.00      1.00        12
         MCL       0.00      0.00      0.00         0

    accuracy                           0.38        39
   macro avg       0.30      0.50      0.33        39
weighted avg       0.32      0.38      0.33        39

Final model trained with 4 classes.

Results on test set:
              precision    recall  f1-score   support

       DLBCL       0.86      0.67      0.75        18
          FL       0.65      0.80      0.72        35
          HL       1.00      1.00      1.00        13
         MCL       0.81      0.72      0.76        36

    accuracy                           0.77       102
   macro avg       0.83      0.80      0.81       102
weighted avg       0.79      0.77      0.78       102
```

## SHAP RESULTS

1. **`shap_bar.png`**: 
   ![shap_bar](results/shap_bar.png)

2. **`shap_summary.png`**:
    ![shap_summary](results/shap_summary.png)

3. **`shap_waterfall.png`**:
    ![shap_waterfall](results/shap_waterfall.png)