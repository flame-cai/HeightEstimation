"""
Results generation for the Research Letter.

Computes:
  - Height prediction performance against stadiometer measurements
  - LHFA z-scores using WHO 2006 Child Growth Standards
  - Stunting risk classification (z < -2)

Inputs:
  Data.csv                 — uuid, gender, age_days, height_cm
  results.csv              — uuid, ensemble
  who_tables/lhfa_*.xlsx   — WHO expanded LMS z-score tables

Outputs:
  height_paper_results.csv — per-child predictions, z-scores, labels
  Console tables           — regression and classification metrics
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from scipy.stats import pearsonr, binomtest

warnings.filterwarnings("ignore")

SEED = 0
N_BOOT = 2000
rng = np.random.default_rng(SEED)

# WHO LHFA tables
WHO_BOYS = pd.read_excel(
    "who_tables/lhfa_boys_expanded.xlsx",
    sheet_name=0,
).set_index("Day")

WHO_GIRLS = pd.read_excel(
    "who_tables/lhfa_girls_expanded.xlsx",
    sheet_name=0,
).set_index("Day")


def lhfa_zscore(sex, age_days, height_cm):
    if pd.isna(age_days) or pd.isna(height_cm):
        return np.nan

    age_days = int(round(age_days))

    if age_days < 0 or age_days > 1856:
        return np.nan

    tbl = WHO_BOYS if str(sex).upper().startswith("M") else WHO_GIRLS

    L = float(tbl.loc[age_days, "L"])
    M = float(tbl.loc[age_days, "M"])
    S = float(tbl.loc[age_days, "S"])

    if L == 0:
        return np.log(height_cm / M) / S

    return ((height_cm / M) ** L - 1) / (L * S)


def classify(z):
    if pd.isna(z):
        return None
    return "At-risk" if z < -2 else "Normal"


# Data load
truth = pd.read_csv("Data.csv")
ht = pd.read_csv("results.csv")

ht["ensemble"] = pd.to_numeric(ht["ensemble"], errors="coerce")

df = ht.merge(truth, on="uuid")

df = df.dropna(
    subset=["ensemble", "height_cm", "age_days", "gender"]
).reset_index(drop=True)

df["pred_h"] = df["ensemble"]
df["age_months"] = df["age_days"] / 30.4375

# WHO Anthro Manual v3.2.2 §3.3:
# Children <24 months are assessed using recumbent length.
# Standing-height measurements are adjusted by +0.7 cm before LHFA lookup.
truth_h = df["height_cm"] + (df["age_days"] < 731) * 0.7

df["true_z"] = [
    lhfa_zscore(s, a, h)
    for s, a, h in zip(
        df["gender"],
        df["age_days"],
        truth_h,
    )
]

df["pred_z"] = [
    lhfa_zscore(s, a, h)
    for s, a, h in zip(
        df["gender"],
        df["age_days"],
        df["pred_h"],
    )
]

df["true_lhfa"] = [classify(z) for z in df["true_z"]]
df["pred_lhfa"] = [classify(z) for z in df["pred_z"]]

df.to_csv("height_paper_results.csv", index=False)


# Helpers
def boot_ci(arrays, fn, B=N_BOOT, alpha=0.05):
    n = len(arrays[0])

    out = []

    for _ in range(B):
        idx = rng.integers(0, n, n)

        try:
            out.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            pass

    out = np.array(out)
    out = out[~np.isnan(out)]

    return np.percentile(
        out,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
    )


def fmt(val, lo, hi, dp=2, pct=False):
    suffix = "%" if pct else ""
    return f"{val:.{dp}f}{suffix} ({lo:.{dp}f}–{hi:.{dp}f})"


# Table 1: Height prediction performance
print("=" * 78)
print(
    "TABLE 1. Photograph-based height vs. stadiometer (n = {})".format(
        len(df)
    )
)
print("=" * 78)

print(
    f"{'Age band':<12} "
    f"{'n':>4} "
    f"{'MAE (cm)':>22} "
    f"{'RMSE (cm)':>22} "
    f"{'MAPE (%)':>10}"
)

print("-" * 78)

bins = [
    ("< 12 mo", df["age_months"] < 12),
    (
        "12-35 mo",
        (df["age_months"] >= 12)
        & (df["age_months"] < 36),
    ),
    ("36+ mo", df["age_months"] >= 36),
    ("Overall", pd.Series([True] * len(df))),
]

for label, mask in bins:
    yt = df.loc[mask, "height_cm"].values
    yp = df.loc[mask, "pred_h"].values

    n = len(yt)

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100

    if n >= 10:
        mae_lo, mae_hi = boot_ci(
            (yt, yp),
            lambda a, b: mean_absolute_error(a, b),
        )

        rmse_lo, rmse_hi = boot_ci(
            (yt, yp),
            lambda a, b: np.sqrt(
                mean_squared_error(a, b)
            ),
        )

        mae_s = fmt(mae, mae_lo, mae_hi)
        rmse_s = fmt(rmse, rmse_lo, rmse_hi)

    else:
        mae_s = f"{mae:.2f}"
        rmse_s = f"{rmse:.2f}"

    print(
        f"{label:<12} "
        f"{n:>4} "
        f"{mae_s:>22} "
        f"{rmse_s:>22} "
        f"{mape:>9.2f}"
    )

yt_all = df["height_cm"].values
yp_all = df["pred_h"].values

r, _ = pearsonr(yt_all, yp_all)

r_lo, r_hi = boot_ci(
    (yt_all, yp_all),
    lambda a, b: pearsonr(a, b)[0],
)

print(
    f"\nOverall Pearson r = "
    f"{r:.3f} ({r_lo:.3f}-{r_hi:.3f})."
)


# Table 2: LHFA classification
print("\n" + "=" * 78)
print(
    "TABLE 2. LHFA binary screening "
    "(At-risk: z < -2 vs Normal: z >= -2)"
)
print("=" * 78)

v = (
    df["true_lhfa"].notna()
    & df["pred_lhfa"].notna()
)

yt = df.loc[v, "true_lhfa"].values
yp = df.loc[v, "pred_lhfa"].values

n = len(yt)

n_atrisk = int((yt == "At-risk").sum())
n_normal = int((yt == "Normal").sum())

baseline = max(n_atrisk, n_normal) / n


def cls_pr(yt, yp, cls):
    yt_b = (yt == cls).astype(int)
    yp_b = (yp == cls).astype(int)

    tp = int(((yt_b == 1) & (yp_b == 1)).sum())
    fp = int(((yt_b == 0) & (yp_b == 1)).sum())
    fn = int(((yt_b == 1) & (yp_b == 0)).sum())

    p = tp / (tp + fp) if (tp + fp) else np.nan
    r = tp / (tp + fn) if (tp + fn) else np.nan

    f1 = (
        2 * p * r / (p + r)
        if (p and r and (p + r))
        else np.nan
    )

    return p, r, f1


print(
    f"{'Group':<22} "
    f"{'n':>4} "
    f"{'Precision':>22} "
    f"{'Recall':>22} "
    f"{'F1':>22}"
)

print("-" * 94)

for cls in ["At-risk", "Normal"]:
    p, r, f = cls_pr(yt, yp, cls)

    p_lo, p_hi = boot_ci(
        (yt, yp),
        lambda a, b: cls_pr(a, b, cls)[0],
    )

    r_lo, r_hi = boot_ci(
        (yt, yp),
        lambda a, b: cls_pr(a, b, cls)[1],
    )

    f_lo, f_hi = boot_ci(
        (yt, yp),
        lambda a, b: cls_pr(a, b, cls)[2],
    )

    sup = int((yt == cls).sum())

    print(
        f"{cls:<22} "
        f"{sup:>4} "
        f"{fmt(p*100, p_lo*100, p_hi*100, dp=1, pct=True):>22} "
        f"{fmt(r*100, r_lo*100, r_hi*100, dp=1, pct=True):>22} "
        f"{fmt(f*100, f_lo*100, f_hi*100, dp=1, pct=True):>22}"
    )

wp = precision_score(
    yt,
    yp,
    average="weighted",
    zero_division=0,
) * 100

wr = recall_score(
    yt,
    yp,
    average="weighted",
    zero_division=0,
) * 100

wf = f1_score(
    yt,
    yp,
    average="weighted",
    zero_division=0,
) * 100

acc = accuracy_score(yt, yp) * 100

wp_ci = boot_ci(
    (yt, yp),
    lambda a, b: precision_score(
        a,
        b,
        average="weighted",
        zero_division=0,
    ) * 100,
)

wr_ci = boot_ci(
    (yt, yp),
    lambda a, b: recall_score(
        a,
        b,
        average="weighted",
        zero_division=0,
    ) * 100,
)

wf_ci = boot_ci(
    (yt, yp),
    lambda a, b: f1_score(
        a,
        b,
        average="weighted",
        zero_division=0,
    ) * 100,
)

acc_ci = boot_ci(
    (yt, yp),
    lambda a, b: accuracy_score(a, b) * 100,
)

print(
    f"{'Overall (weighted)':<22} "
    f"{n:>4} "
    f"{fmt(wp, wp_ci[0], wp_ci[1], dp=1, pct=True):>22} "
    f"{fmt(wr, wr_ci[0], wr_ci[1], dp=1, pct=True):>22} "
    f"{fmt(wf, wf_ci[0], wf_ci[1], dp=1, pct=True):>22}"
)

kappa = cohen_kappa_score(yt, yp)

kap_ci = boot_ci(
    (yt, yp),
    lambda a, b: cohen_kappa_score(a, b),
)

correct = int((yt == yp).sum())

bt = binomtest(
    correct,
    n,
    p=baseline,
    alternative="greater",
)

print(
    f"\nOverall accuracy   = "
    f"{fmt(acc, acc_ci[0], acc_ci[1], dp=1, pct=True)}"
)

print(
    f"Cohen's kappa      = "
    f"{kappa:.3f} ({kap_ci[0]:.3f}-{kap_ci[1]:.3f})"
)

print(
    f"Majority baseline  = "
    f"{baseline*100:.1f}%"
)

print(
    f"Binomial test p    = "
    f"{bt.pvalue:.3g} "
    f"(one-sided, accuracy > baseline)"
)

print("\nWrote height_paper_results.csv")

