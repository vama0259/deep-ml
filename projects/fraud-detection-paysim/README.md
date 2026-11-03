# Fraud detection on PaySim

A FinCrime-style project: find fraudulent mobile-money transactions in the PaySim dataset
([Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1), about 6.3M rows, 0.13% fraud).

## What it shows
- **Data checks first:** nulls, duplicates, fraud rate, and which transaction types contain fraud (only `TRANSFER` and `CASH_OUT`).
- **No leakage:** a time-based train, validation and test split instead of a random one, and `isFlaggedFraud` dropped.
- **Domain features:** balance errors at origin and destination, "account emptied", destination was empty.
- **Right metric:** PR-AUC, not accuracy or ROC-AUC, because fraud is rare.
- **Models:** logistic regression baseline, LightGBM main model, isolation forest for the no-labels case.
- **Business threshold:** chosen on validation by cost (missed fraud amount + review fee), applied once to the test set.

## Run it
- **Colab:** open `fraud_detection.ipynb`, run all. It downloads the data with `kagglehub` and asks you to log in to Kaggle.
- **Locally:** `pip install -r requirements.txt`, put the PaySim CSV here (or set `DATA_PATH`), then `python fraud_detection.py`.

## Results
Fill in after running on the real data (test PR-AUC, precision, recall at the chosen threshold, cost vs flag-nothing).

## Limits
PaySim is simulated, so real fraud is messier. No per-customer history features yet (needs point-in-time windows).
Review cost is an assumption. Next steps: SHAP explanations, calibration, drift monitoring.
