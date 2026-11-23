# Missing-Data Imputation Benchmark

Compares five ways of filling missing values (plus one that doesn't fill them at all) under three missingness mechanisms. Each method is scored on two different questions:

1. **How close are the filled values to the truth?** RMSE on the hidden cells, in standard-deviation units.
2. **How well does a model trained on the imputed data predict?** Test ROC-AUC.

## Setup

- Breast-cancer data: 30 correlated measurements. Missing values are injected into 12 of the columns at 10% and 30%.
- Mechanisms:
  - **MCAR**: every cell has the same chance of going missing
  - **MAR**: missingness depends on another observed column
  - **MNAR**: large values are more likely to be missing, like high earners skipping an income question
- Imputers are fit on the training split only. Results are averaged over 5 random splits.
- Methods: mean, median, mean + missing-indicator columns, KNN (k=5), iterative (MICE-style regression on the other columns), and HistGradientBoosting with native NaN handling (no imputation).

## Results (30% missing)

| Method | MCAR RMSE | MAR RMSE | MNAR RMSE | AUC range |
|---|---|---|---|---|
| mean | 0.96 | 1.19 | 1.34 | 0.991–0.993 |
| median | 0.98 | 1.28 | 1.45 | 0.992–0.993 |
| KNN (k=5) | 0.53 | 0.66 | 0.76 | 0.991–0.993 |
| **iterative** | **0.41** | **0.46** | **0.48** | 0.991–0.993 |
| HistGB native NaN | – | – | – | 0.987–0.988 |
| *complete data* | 0 | 0 | 0 | 0.993 |

## What this shows

- **Iterative imputation recovers values 2–3× more accurately than mean filling.** The features are strongly correlated, so the other columns predict the missing one well.
- **Mean and median get worse under MAR and MNAR**, because the values that went missing differ systematically from the ones left (in MNAR they're the large ones). The column average is then the wrong guess. Model-based methods hold up much better.
- **But the downstream AUC barely changes.** Every imputer stays within 0.002 of the complete-data AUC. With 30 redundant features, the classifier gets the signal from the columns that aren't missing. **Better imputation only pays off when the missing feature carries information nothing else does.** Measure both; don't assume one follows from the other.
- **HistGB's lower AUC isn't caused by the missing values.** It scores the same with complete data, because logistic regression simply suits this dataset better. Native NaN handling is still the simplest option when you're using trees anyway.

## How to run

```bash
pip install -r requirements.txt
python benchmark.py     # ~2 minutes
pytest
```

## Next steps
- Repeat on a dataset where one important feature goes missing
- Add multiple imputation, pooling estimates across several imputed copies
