# %% [markdown]
# # Fraud detection on PaySim (FinCrime project)
# Mobile-money transactions with a ~0.13% fraud rate. The goal is a model a fraud team could use:
# honest time-based validation, PR-AUC instead of accuracy, a threshold picked by cost, and a
# comparison with an unsupervised detector for the case where labels are missing.

# %% [markdown]
# ## 1. Load the data
# On Colab this downloads PaySim from Kaggle with `kagglehub` (it asks you to log in once).
# Outside Colab, put `PS_*.csv` next to this file or set `DATA_PATH`.

# %%
import os, glob
import numpy as np
import pandas as pd

def load_paysim():
    path = os.environ.get("DATA_PATH")
    if path is None:
        local = glob.glob("PS_*.csv") + glob.glob("*paysim*.csv")
        if local:
            path = local[0]
        else:
            import kagglehub
            folder = kagglehub.dataset_download("ealaxi/paysim1")
            path = glob.glob(os.path.join(folder, "*.csv"))[0]
    return pd.read_csv(path)

df = load_paysim()
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Data quality and first look
# Check size, nulls, duplicates, the fraud rate and which transaction types contain fraud at all.

# %%
print(df.isna().sum().sum(), "nulls |", df.duplicated().sum(), "duplicate rows")
print(f"fraud rate: {df.isFraud.mean():.4%} ({df.isFraud.sum()} of {len(df)})")
print(df.groupby("type").isFraud.agg(["count", "sum"]))

# %% [markdown]
# Fraud only happens in `TRANSFER` and `CASH_OUT`, so only those rows are modelled. The other
# types are never fraud, so scoring them would just inflate the numbers.
#
# `isFlaggedFraud` (the simulator's own rule) is dropped: it is nearly always 0 and would be a leak.

# %%
df = df[df.type.isin(["TRANSFER", "CASH_OUT"])].drop(columns=["isFlaggedFraud"]).copy()
print(df.shape, f"fraud rate in scope: {df.isFraud.mean():.3%}")

# %% [markdown]
# ## 3. Features
# Balance errors are the strong signal: in fraud the amount often empties the origin account, and the
# books at origin and destination often do not add up. Names are IDs, so they are not used directly.

# %%
def make_features(d):
    out = pd.DataFrame(index=d.index)
    out["step"] = d["step"]
    out["hour"] = d["step"] % 24
    out["is_transfer"] = (d["type"] == "TRANSFER").astype(int)
    out["amount"] = d["amount"]
    out["log_amount"] = np.log1p(d["amount"])
    out["oldbalanceOrg"] = d["oldbalanceOrg"]
    out["newbalanceOrig"] = d["newbalanceOrig"]
    out["oldbalanceDest"] = d["oldbalanceDest"]
    out["newbalanceDest"] = d["newbalanceDest"]
    out["err_orig"] = d["oldbalanceOrg"] - d["amount"] - d["newbalanceOrig"]
    out["err_dest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    out["empties_origin"] = ((d["oldbalanceOrg"] > 0) & (d["newbalanceOrig"] == 0)).astype(int)
    out["dest_was_empty"] = (d["oldbalanceDest"] == 0).astype(int)
    return out

X = make_features(df)
y = df["isFraud"]

# %% [markdown]
# ## 4. Time-based split
# A random split would let the model see the future. Train on the first 70% of the time steps,
# validate on the next 15% (threshold choice) and test on the last 15% (reported once).

# %%
q1, q2 = df["step"].quantile([0.70, 0.85])
train, val, test = df.step <= q1, (df.step > q1) & (df.step <= q2), df.step > q2
for name, m in [("train", train), ("val", val), ("test", test)]:
    print(f"{name}: {m.sum():>8} rows, {y[m].sum():>5} frauds")

# %% [markdown]
# ## 5. Baselines and the main model
# Logistic regression is the simple baseline. LightGBM is the main model. `scale_pos_weight`
# handles the imbalance without resampling the data.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb

drop_time = ["step"]   # step is for splitting, not a feature that generalises
F = [c for c in X.columns if c not in drop_time]

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
logit.fit(X.loc[train, F], y[train])

pos_w = (y[train] == 0).sum() / max((y[train] == 1).sum(), 1)
gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         scale_pos_weight=pos_w, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=0, verbose=-1)
gbm.fit(X.loc[train, F], y[train])

scores = {"logistic": logit.predict_proba(X.loc[val, F])[:, 1],
          "lightgbm": gbm.predict_proba(X.loc[val, F])[:, 1]}
for k, s in scores.items():
    print(f"{k:9s} val PR-AUC {average_precision_score(y[val], s):.3f} | ROC-AUC {roc_auc_score(y[val], s):.3f}")

# %% [markdown]
# PR-AUC is the main metric. With 0.1 to 1% fraud, ROC-AUC looks great even for weak models
# because the huge number of true negatives hides false alarms.

# %% [markdown]
# ## 6. Pick the threshold by cost
# Assume a missed fraud costs its amount and a false alarm costs a fixed review fee. Choose the
# threshold on the validation set, then apply it unchanged to the test set.

# %%
REVIEW_COST = 5.0

def total_cost(y_true, score, amount, thr):
    flag = score >= thr
    missed = amount[(~flag) & (y_true == 1)].sum()
    reviews = flag.sum() * REVIEW_COST
    return missed + reviews

amt_val = df.loc[val, "amount"].values
yv = y[val].values
sv = scores["lightgbm"]
thrs = np.unique(np.quantile(sv, np.linspace(0.5, 0.9999, 200)))
costs = [total_cost(yv, sv, amt_val, t) for t in thrs]
best_thr = thrs[int(np.argmin(costs))]
print(f"best threshold {best_thr:.3f}  cost {min(costs):,.0f}  vs flag-nothing {amt_val[yv == 1].sum():,.0f}")

# %% [markdown]
# ## 7. Final test result (looked at once)

# %%
from sklearn.metrics import precision_score, recall_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt

st = gbm.predict_proba(X.loc[test, F])[:, 1]
yt, amt_t = y[test].values, df.loc[test, "amount"].values
flag = st >= best_thr
print(f"test PR-AUC {average_precision_score(yt, st):.3f} | ROC-AUC {roc_auc_score(yt, st):.3f}")
print(f"at the chosen threshold: precision {precision_score(yt, flag, zero_division=0):.3f}, "
      f"recall {recall_score(yt, flag):.3f}, flagged {flag.sum()} of {len(yt)}")
print(f"cost {total_cost(yt, st, amt_t, best_thr):,.0f} vs flag-nothing {amt_t[yt == 1].sum():,.0f}")
PrecisionRecallDisplay.from_predictions(yt, st)
plt.title("Test precision-recall (LightGBM)"); plt.show()

# %% [markdown]
# ## 8. What the model uses

# %%
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=F).sort_values(ascending=False)
imp.head(8).plot.barh().invert_yaxis(); plt.title("LightGBM gain importance"); plt.show()

# %% [markdown]
# ## 9. When there are no labels: isolation forest
# Many fraud teams have few or late labels. Isolation forest scores how easy a row is to isolate,
# with no labels. Compare it on the same test rows.

# %%
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0)
iso.fit(X.loc[train, F])
s_iso = -iso.score_samples(X.loc[test, F])
print(f"isolation forest test PR-AUC {average_precision_score(yt, s_iso):.3f} "
      f"(supervised LightGBM {average_precision_score(yt, st):.3f})")

# %% [markdown]
# ## 10. Limits and next steps
# - PaySim is simulated, so real fraud will be messier and the balance-error signal weaker.
# - No customer history features (counts or amounts in the last 24h per account) are used. These are the
#   usual next step and must be computed with a point-in-time window to avoid leakage.
# - Cost numbers are assumptions. In production the review cost and the fraud loss would come from the team.
# - Next: SHAP for explanations, calibration, and drift monitoring.
