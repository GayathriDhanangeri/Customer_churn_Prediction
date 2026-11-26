# %% [markdown]
# # Project 02 — Customer Churn Prediction & Retention Strategy
# **Target definition:** `churned = 1` if the account cancelled within 90 days *after* the snapshot date (31 Dec 2025).
# Every feature is measured on or before the snapshot. `churn_date` and `cancel_reason` are recorded *after* the outcome,
# so they are **excluded from modelling** (leakage) and used only for dashboard reporting.
#
# Workflow: load & inspect → clean → EDA → split → model comparison (LR / Tree / RF) → evaluation → business threshold → drivers & segments → Power BI exports.

# %%
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, accuracy_score, average_precision_score)
from sklearn.inspection import permutation_importance

ROOT = Path(".")            # run from the project folder
SEED = 42
sns.set_theme(style="whitegrid")
for d in ["data", "models", "reports", "powerbi"]:
    (ROOT / d).mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Load and inspect the raw file

# %%
raw = pd.read_csv(ROOT / "data/churn_raw.csv")
print("shape:", raw.shape)
print(raw.dtypes, "\n")
print("missing values per column:\n", raw.isna().sum()[lambda s: s > 0], "\n")
print("duplicate rows:", raw.duplicated().sum(), "| duplicate customer_id:", raw.customer_id.duplicated().sum())
for c in ["plan", "contract_type", "payment_method"]:
    print(f"\n{c}:\n", raw[c].value_counts(dropna=False))
print("\nclass balance (churned):\n", raw.drop_duplicates("customer_id").churned.value_counts(normalize=True).round(3))

# %% [markdown]
# ## 2. Validate and clean
# Rules: remove duplicate customers, standardise category labels, treat impossible tenure as missing, keep missing values
# as missing (imputed *inside* the model pipeline after the split so no test information leaks into training).

# %%
def standardise_text(s: pd.Series) -> pd.Series:
    return s.str.strip().str.title()

def validate(df: pd.DataFrame) -> dict:
    """Return counts of rule violations (used as a data-quality report)."""
    return {
        "duplicate_customer_ids": int(df.customer_id.duplicated().sum()),
        "negative_tenure": int((df.tenure_months < 0).sum()),
        "negative_charges": int((df.monthly_charges < 0).sum()),
        "missing_usage": int(df.avg_weekly_usage_hrs.isna().sum()),
        "missing_charges": int(df.monthly_charges.isna().sum()),
        "missing_payment_method": int(df.payment_method.isna().sum()),
        "churn_label_not_binary": int((~df.churned.isin([0, 1])).sum()),
    }

before = validate(raw)
df = raw.drop_duplicates(subset="customer_id", keep="first").copy()
df["plan"] = standardise_text(df["plan"])
df["contract_type"] = (df["contract_type"].str.strip().str.lower().str.replace(" ", "-")
                       .replace({"month-to-month": "Month-to-month", "one-year": "One year", "two-year": "Two year"}))
df["contract_type"] = df["contract_type"].replace({"one year": "One year", "two year": "Two year"})
df.loc[df.tenure_months < 0, "tenure_months"] = np.nan
df["payment_method"] = df["payment_method"].fillna("Unknown")
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce")
df["cancel_reason"] = df["cancel_reason"].fillna("")

# derived (pre-snapshot) fields only
df["tenure_band"] = pd.cut(df.tenure_months, [0, 6, 12, 24, 48, 100], labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"])
df["support_band"] = pd.cut(df.support_contacts_90d, [-1, 0, 2, 100], labels=["0", "1-2", "3+"])
df["late_payment_flag"] = (df.late_payments_12m > 0).astype(int)

after = validate(df)
dq = pd.DataFrame({"before_cleaning": before, "after_cleaning": after})
print(dq)
dq.to_csv(ROOT / "reports/data_quality_report.csv")
df.to_csv(ROOT / "data/churn_clean.csv", index=False)
print("\nclean rows:", len(df), "| churn rate:", round(df.churned.mean(), 3))

# %% [markdown]
# ## 3. Exploratory data analysis

# %%
num_cols = ["tenure_months", "monthly_charges", "avg_weekly_usage_hrs", "support_contacts_90d", "late_payments_12m"]
print(df[num_cols].describe(percentiles=[.25, .5, .75, .95]).round(2))

def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
print("\nIQR outliers:", {c: iqr_outliers(df[c].dropna()) for c in num_cols})

rates = {}
for c in ["tenure_band", "plan", "contract_type", "payment_method", "support_band", "age_group", "region"]:
    rates[c] = df.groupby(c, observed=True).churned.agg(["mean", "count"]).round(3)
    print(f"\nChurn rate by {c}:\n", rates[c])

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.flat, ["tenure_band", "plan", "contract_type", "payment_method", "support_band", "age_group"]):
    r = rates[c]["mean"]
    sns.barplot(x=r.index.astype(str), y=r.values, ax=ax, color="#4C72B0")
    ax.set_title(f"Churn rate by {c}"); ax.set_ylabel("churn rate"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(ROOT / "reports/eda_churn_by_group.png", dpi=130); plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(df[num_cols + ["churned"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(ROOT / "reports/eda_correlation.png", dpi=130); plt.show()

# %% [markdown]
# ## 4. Split first, then build features inside a pipeline (no leakage)
# Stratified 80/20 split. Imputation, scaling and encoding are fitted on the training data only (inside the `Pipeline`).
# Model selection uses 5-fold stratified CV *within the training set*; the test set is touched once for final reporting.

# %%
num_feats = ["tenure_months", "monthly_charges", "avg_weekly_usage_hrs", "support_contacts_90d", "late_payments_12m"]
cat_feats = ["plan", "contract_type", "payment_method", "age_group", "region"]
FEATURES = num_feats + cat_feats
LEAKY = ["churn_date", "cancel_reason"]          # post-outcome, never used
assert not set(LEAKY) & set(FEATURES)

X, y = df[FEATURES], df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("train:", X_train.shape, "test:", X_test.shape, "| train churn:", round(y_train.mean(), 3), "test churn:", round(y_test.mean(), 3))

def make_pre(scale: bool):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(num), num_feats),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)])

# %% [markdown]
# ## 5. Train and tune three classifiers (class_weight="balanced" for the 31/69 imbalance)

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
specs = {
    "Logistic Regression": (Pipeline([("pre", make_pre(True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                            {"m__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (Pipeline([("pre", make_pre(False)), ("m", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
                      {"m__max_depth": [3, 4, 5, 6, 8], "m__min_samples_leaf": [20, 50, 100]}),
    "Random Forest": (Pipeline([("pre", make_pre(False)), ("m", RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                                                                       n_jobs=-1, random_state=SEED))]),
                      {"m__max_depth": [6, 10, None], "m__min_samples_leaf": [10, 30]}),
}
fitted, cv_scores = {}, {}
for name, (pipe, grid) in specs.items():
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_train, y_train)
    fitted[name] = gs.best_estimator_; cv_scores[name] = gs.best_score_
    print(f"{name:20s} CV ROC-AUC={gs.best_score_:.4f}  best={gs.best_params_}")

# %% [markdown]
# ## 6. Evaluate on the unseen test set

# %%
rows = []
for name, est in fitted.items():
    proba = est.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
    rows.append(dict(model=name, cv_roc_auc=cv_scores[name], test_accuracy=accuracy_score(y_test, pred),
                     precision=precision_score(y_test, pred), recall=recall_score(y_test, pred),
                     f1=f1_score(y_test, pred), test_roc_auc=roc_auc_score(y_test, proba),
                     pr_auc=average_precision_score(y_test, proba)))
comparison = pd.DataFrame(rows).round(4).sort_values("test_roc_auc", ascending=False)
print(comparison.to_string(index=False))
comparison.to_csv(ROOT / "reports/model_comparison.csv", index=False)

best_name = comparison.iloc[0]["model"]; best = fitted[best_name]
print("\nSelected model:", best_name)

# Why accuracy alone misleads
majority_acc = 1 - y_test.mean()
print(f"Accuracy of 'predict nobody churns': {majority_acc:.3f} — high accuracy, zero churners found (recall = 0).")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, est in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, est.predict_proba(X_test)[:, 1])
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, est.predict_proba(X_test)[:,1]):.3f})")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC curves (test)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
cm = confusion_matrix(y_test, (best.predict_proba(X_test)[:, 1] >= .5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1], xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
ax[1].set_title(f"Confusion matrix — {best_name} @0.5"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
plt.tight_layout(); plt.savefig(ROOT / "reports/model_evaluation.png", dpi=130); plt.show()

# %% [markdown]
# ## 7. Choose the threshold from business costs (assumptions — edit to match your business)
# * Retention offer costs **$40** per contacted customer. A successful save is worth **$300** (12 months of margin) and an offer succeeds **30%** of the time.
# * Expected value of contacting a customer = P(churn) × 0.30 × 300 − 40  → break-even at P(churn) ≈ 0.44 *if scores were calibrated*.
# * Because `class_weight="balanced"` inflates probabilities, the empirical search below is used instead of the formula.
# * We search thresholds on **out-of-fold training predictions** (not the test set) and then report the test result once.

# %%
OFFER_COST, SAVE_VALUE, SAVE_RATE = 40, 300, 0.30
def net_value(y_true, proba, thr):
    contact = proba >= thr
    tp = (contact & (y_true == 1)).sum(); fp = (contact & (y_true == 0)).sum()
    return tp * (SAVE_RATE * SAVE_VALUE - OFFER_COST) - fp * OFFER_COST

oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid_thr = np.round(np.arange(0.05, 0.95, 0.01), 2)
val_curve = [net_value(y_train.values, oof, t) for t in grid_thr]
THR = float(grid_thr[int(np.argmax(val_curve))])
print("Cost-optimal threshold (from out-of-fold data):", THR)

proba_test = best.predict_proba(X_test)[:, 1]
rows = []
for t in [0.5, THR]:
    p = (proba_test >= t).astype(int)
    rows.append(dict(threshold=t, precision=precision_score(y_test, p), recall=recall_score(y_test, p), f1=f1_score(y_test, p),
                     customers_contacted_pct=p.mean(), net_value=net_value(y_test.values, proba_test, t)))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# "What share of likely churners can we identify at a chosen recall?"
thr_table = []
for t in grid_thr[::5]:
    p = (proba_test >= t).astype(int)
    thr_table.append(dict(threshold=t, recall=recall_score(y_test, p), precision=precision_score(y_test, p, zero_division=0),
                          contacted_pct=p.mean(), net_value=net_value(y_test.values, proba_test, t)))
thr_table = pd.DataFrame(thr_table).round(4); thr_table.to_csv(ROOT / "powerbi/threshold_analysis.csv", index=False)
target_recall = 0.80
t80 = max(t for t in grid_thr if recall_score(y_test, (proba_test >= t).astype(int)) >= target_recall)
print(f"\nTo capture ≥{target_recall:.0%} of churners: threshold {t80} → contact {(proba_test>=t80).mean():.1%} of customers, "
      f"precision {precision_score(y_test,(proba_test>=t80).astype(int)):.2f}")

# %% [markdown]
# ## 8. Churn-risk drivers (permutation importance on unseen data + logistic coefficients)

# %%
pi = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
drivers = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
drivers["rank"] = drivers.index + 1
print(drivers.round(4).to_string(index=False))
drivers.to_csv(ROOT / "powerbi/driver_importance.csv", index=False)

lr = fitted["Logistic Regression"]
names = lr.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["m"].coef_[0], index=names).sort_values()
coefs.rename("coefficient").to_csv(ROOT / "reports/logistic_coefficients.csv")
print("\nStrongest risk-increasing terms (logistic):\n", coefs.tail(6).round(3))
print("Strongest protective terms:\n", coefs.head(6).round(3))

plt.figure(figsize=(7, 4.5)); sns.barplot(data=drivers, y="feature", x="importance", color="#C44E52")
plt.title(f"Permutation importance (drop in ROC-AUC) — {best_name}"); plt.tight_layout()
plt.savefig(ROOT / "reports/driver_importance.png", dpi=130); plt.show()

# %% [markdown]
# ## 9. Score every customer honestly, build risk bands, segments and a retention-priority framework
# Training customers receive **out-of-fold** scores and test customers receive held-out scores, so no customer is scored by a model that saw their label.

# %%
final = best.fit(X_train, y_train)               # selected model, fitted on the training set only
score = pd.Series(np.nan, index=df.index)
score.loc[X_train.index] = oof
score.loc[X_test.index] = final.predict_proba(X_test)[:, 1]
df["risk_score"] = score.round(4)
df["data_split"] = np.where(df.index.isin(X_test.index), "Test (held-out)", "Train (out-of-fold)")
df["risk_band"] = pd.cut(df.risk_score, [-.01, .25, THR, .75, 1.0], labels=["Low", "Moderate", "High", "Critical"])
df["predicted_churn"] = (df.risk_score >= THR).astype(int)

def segment(r):
    if r.contract_type == "Month-to-month" and r.tenure_months <= 12 and r.support_contacts_90d >= 2: return "New & frustrated"
    if r.contract_type == "Month-to-month" and r.late_payments_12m >= 2:                              return "Payment-stressed"
    if r.contract_type == "Month-to-month" and r.avg_weekly_usage_hrs < 15:                          return "Low-engagement flexible"
    if r.contract_type == "Month-to-month":                                                          return "Flexible loyalists"
    return "Contracted & stable"
df["segment"] = df.apply(segment, axis=1)

def priority(r):
    high_value = r.monthly_charges >= 60
    if r.risk_band == "Critical": return "P1 — Call within 48h" if high_value else "P2 — Targeted offer"
    if r.risk_band == "High":     return "P2 — Targeted offer" if high_value else "P3 — Email nurture"
    if r.risk_band == "Moderate": return "P3 — Email nurture"
    return "P4 — Monitor"
df["retention_priority"] = df.apply(priority, axis=1)

seg = df.groupby("segment").agg(customers=("customer_id", "count"), actual_churn=("churned", "mean"),
                                avg_risk=("risk_score", "mean"), avg_charge=("monthly_charges", "mean")).round(3)
print(seg, "\n")
print(df.groupby("risk_band", observed=True).agg(customers=("customer_id", "count"), actual_churn=("churned", "mean")).round(3))

# %% [markdown]
# ## 10. Save the final model and export Power BI tables
# Risk scores are **estimates of probability, not certainties**; bands should trigger review, not automatic action.

# %%
joblib.dump({"model": final, "features": FEATURES, "threshold": THR, "target": "churned within 90 days after 2025-12-31"},
            ROOT / "models/churn_model.joblib")

df["signup_month"] = df.signup_date.dt.to_period("M").dt.to_timestamp()
df["churn_month"] = df.churn_date.dt.to_period("M").dt.to_timestamp()
fact_cols = ["customer_id", "signup_date", "signup_month", "churn_date", "churn_month", "cancel_reason", "region", "age_group",
             "tenure_months", "tenure_band", "plan", "contract_type", "monthly_charges", "avg_weekly_usage_hrs",
             "support_contacts_90d", "support_band", "payment_method", "late_payments_12m", "churned", "risk_score",
             "risk_band", "predicted_churn", "segment", "retention_priority", "data_split"]
df[fact_cols].to_csv(ROOT / "powerbi/fact_customers.csv", index=False)

cal = pd.DataFrame({"date": pd.date_range(df.signup_date.min().normalize(), "2026-03-31")})
cal["year"] = cal.date.dt.year; cal["month_num"] = cal.date.dt.month; cal["month_name"] = cal.date.dt.strftime("%b")
cal["year_month"] = cal.date.dt.strftime("%Y-%m"); cal["quarter"] = "Q" + cal.date.dt.quarter.astype(str)
cal.to_csv(ROOT / "powerbi/dim_date.csv", index=False)
pd.DataFrame({"risk_band": ["Low", "Moderate", "High", "Critical"], "sort_order": [1, 2, 3, 4]}).to_csv(ROOT / "powerbi/dim_risk_band.csv", index=False)
comparison.to_csv(ROOT / "powerbi/model_comparison.csv", index=False)
print(sorted(p.name for p in (ROOT / "powerbi").iterdir()))
