"""Synthetic subscription-customer dataset for Project 02 (churn).
Target: churned = 1 if the account cancelled in the 90 days AFTER the snapshot date (2025-12-31).
All features are measured on or before the snapshot. churn_date / cancel_reason are post-outcome
and must NOT be used as features.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 10_000
SNAPSHOT = pd.Timestamp("2025-12-31")

tenure = np.clip(rng.gamma(2.2, 14, N), 1, 72).round().astype(int)
signup = SNAPSHOT - pd.to_timedelta(tenure * 30, unit="D")
plan = rng.choice(["Basic", "Standard", "Premium"], N, p=[.40, .38, .22])
contract = np.where(rng.random(N) < np.where(tenure > 24, .55, .30),
                    rng.choice(["One year", "Two year"], N, p=[.6, .4]), "Month-to-month")
base = {"Basic": 25, "Standard": 48, "Premium": 80}
monthly = np.array([base[p] for p in plan]) + rng.normal(0, 6, N)
monthly = monthly.clip(10).round(2)
usage = (rng.gamma(2.5, 9, N) * np.where(plan == "Premium", 1.4, np.where(plan == "Basic", .7, 1))).round(1)
support = rng.poisson(np.where(contract == "Month-to-month", 1.6, .8))
payment = rng.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], N, p=[.30, .27, .28, .15])
late = rng.poisson(np.where(payment == "Electronic check", 1.3, .5))
region = rng.choice(["North", "South", "East", "West", "Central"], N, p=[.2, .22, .2, .23, .15])
age = rng.choice(["18-25", "26-35", "36-50", "51+"], N, p=[.15, .32, .33, .20])

z = (-1.4
     - 0.045 * tenure
     + 1.35 * (contract == "Month-to-month")
     - 0.9 * (contract == "Two year")
     + 0.018 * (monthly - 45)
     - 0.022 * (usage - 22)
     + 0.32 * support
     + 0.38 * late
     + 0.55 * (payment == "Electronic check")
     + 0.25 * (age == "18-25")
     + rng.normal(0, .6, N))
p = 1 / (1 + np.exp(-z))
churn = (rng.random(N) < p).astype(int)
churn_date = pd.Series(pd.NaT, index=range(N), dtype="datetime64[ns]")
off = rng.integers(1, 91, N)
churn_date[churn == 1] = SNAPSHOT + pd.to_timedelta(off[churn == 1], unit="D")
reason = np.where(churn == 1, rng.choice(["Price", "Competitor", "Service quality", "No longer needed", "Other"], N,
                  p=[.32, .24, .2, .14, .10]), "")

df = pd.DataFrame(dict(customer_id=[f"C{100000+i}" for i in range(N)], signup_date=signup.date if hasattr(signup,'date') else signup,
    region=region, age_group=age, tenure_months=tenure, plan=plan, contract_type=contract, monthly_charges=monthly,
    avg_weekly_usage_hrs=usage, support_contacts_90d=support, payment_method=payment, late_payments_12m=late,
    churned=churn, churn_date=churn_date, cancel_reason=reason))
df["signup_date"] = pd.to_datetime(signup).strftime("%Y-%m-%d")

# ---- inject realistic data-quality problems (raw file) ----
raw = df.copy()
raw.loc[rng.choice(N, 180, replace=False), "avg_weekly_usage_hrs"] = np.nan
raw.loc[rng.choice(N, 120, replace=False), "monthly_charges"] = np.nan
raw.loc[rng.choice(N, 90, replace=False), "payment_method"] = np.nan
raw.loc[rng.choice(N, 60, replace=False), "plan"] = rng.choice(["basic", "PREMIUM ", "standard"], 60)
raw.loc[rng.choice(N, 40, replace=False), "contract_type"] = "month to month"
raw.loc[rng.choice(N, 15, replace=False), "tenure_months"] = -1          # impossible
raw = pd.concat([raw, raw.sample(75, random_state=1)], ignore_index=True)  # duplicates
raw.to_csv("/home/claude/churn/data/churn_raw.csv", index=False)
print(raw.shape, "raw rows;", round(df.churned.mean(), 3), "churn rate")
