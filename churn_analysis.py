
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# 1. SETTINGS AND DATA LOADING
# ============================================================

FILE = Path(r"D:\HADiL\miras\subscriptions-churn.csv")
OUTPUT = FILE.parent / "churn_regression_results.csv"

TARGET = "ألغى الاشتراك"
ATTENDANCE = "حضر التهيئة"

NUMERIC = [
    "السعر الشهري",
    "أشهر البقاء",
    "تذاكر الدعم",
    "متوسط الجلسات الشهرية",
    "نسبة الخصم"
]

CATEGORICAL = [
    "الباقة",
    "دورة الفوترة",
    "وسيلة الدفع",
    ATTENDANCE
]

REQUIRED = [
    "معرف المشترك",
    TARGET,
    *NUMERIC,
    *CATEGORICAL
]

if not FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE}")

try:
    df = pd.read_csv(FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(FILE, encoding="cp1256")

df.columns = df.columns.str.strip()

missing_columns = [c for c in REQUIRED if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("\n========== 1. DATA OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Duplicate subscriber IDs:",
      int(df["معرف المشترك"].duplicated().sum()))

# ============================================================
# 2. CLEAN AND VALIDATE THE TARGET
# ============================================================

def normalize(value):
    return str(value).strip().lower()

positive_values = {
    "نعم", "yes", "y", "true", "1",
    "ألغى", "ملغى", "منسحب", "انسحب", "churned"
}

negative_values = {
    "لا", "no", "n", "false", "0",
    "لم يلغ", "مستمر", "غير منسحب", "retained", "not churned"
}

target_map = {}
for value in df[TARGET].dropna().unique():
    value_normalized = normalize(value)

    if value_normalized in positive_values:
        target_map[value] = 1
    elif value_normalized in negative_values:
        target_map[value] = 0

df["_churn"] = df[TARGET].map(target_map)

if df["_churn"].isna().any():
    unmapped = df.loc[df["_churn"].isna(), TARGET].unique()
    raise ValueError(f"Unrecognized or missing target values: {unmapped}")

df["_churn"] = df["_churn"].astype(int)

print("\n========== 2. CHURN RATE ==========")
target_counts = df["_churn"].value_counts().sort_index()
print("Retained (0):", int((df["_churn"] == 0).sum()))
print("Churned (1):", int((df["_churn"] == 1).sum()))
print("Churn rate: {:.2f}%".format(df["_churn"].mean() * 100))

# ============================================================
# 3. DATA QUALITY AND DESCRIPTIVE ANALYSIS
# ============================================================

print("\n========== 3. MISSING VALUES ==========")
print(df[REQUIRED].isna().sum().to_string())

print("\n========== 4. NUMERIC COMPARISON ==========")
print(
    df.groupby("_churn")[NUMERIC]
      .mean()
      .T.rename(columns={0: "Retained", 1: "Churned"})
      .round(2)
      .to_string()
)

print("\n========== 5. CHURN RATE BY CATEGORY ==========")
for col in CATEGORICAL:
    temp = df[[col, "_churn"]].copy()
    temp[col] = temp[col].fillna("Missing").astype(str).str.strip()

    summary = temp.groupby(col, dropna=False)["_churn"].agg(
        Customers="size",
        Churned="sum",
        Churn_Rate="mean"
    )

    summary["Churn_Rate"] = (summary["Churn_Rate"] * 100).round(2)
    print(f"\n-- {col} --")
    print(summary.to_string())

print("\n========== 6. PRICE CHECK ==========")
print("\nPrice by churn status:")
print(df.groupby("_churn")["السعر الشهري"].describe().round(2).to_string())

print("\nPrice by plan:")
print(df.groupby("الباقة", dropna=False)["السعر الشهري"]
        .describe().round(2).to_string())

# ============================================================
# 4. PREPARE ANALYSIS DATA
# ============================================================

# "أشهر البقاء" is excluded from predictors because its timing
# relative to cancellation is uncertain and may cause leakage.
#
# Missing numeric predictor values are imputed using the median.
# Missing categorical predictor values are treated as a category.
# This avoids silently deleting many records because of predictors.

PREDICTOR_NUMERIC = [
    "السعر الشهري",
    "تذاكر الدعم",
    "متوسط الجلسات الشهرية",
    "نسبة الخصم"
]

PREDICTOR_CATEGORICAL = CATEGORICAL.copy()

model_df = df[["_churn"] + PREDICTOR_NUMERIC +
              PREDICTOR_CATEGORICAL].copy()

for col in PREDICTOR_NUMERIC:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

for col in PREDICTOR_CATEGORICAL:
    model_df[col] = (
        model_df[col].fillna("Missing").astype(str).str.strip()
    )
    model_df.loc[model_df[col] == "", col] = "Missing"

# Rows with missing target are already rejected above.
# Remove rows with no usable values in every numeric predictor.
all_numeric_missing = model_df[PREDICTOR_NUMERIC].isna().all(axis=1)
model_df = model_df.loc[~all_numeric_missing].copy()

print("\n========== 7. MODEL DATA ==========")
print("Original rows:", len(df))
print("Rows available for modeling:", len(model_df))
print("Rows excluded because all numeric predictors were missing:",
      int(all_numeric_missing.sum()))
print("Churn rate in model data: {:.2f}%".format(
    model_df["_churn"].mean() * 100
))
print("Numeric missing values handled by median imputation:")
print(model_df[PREDICTOR_NUMERIC].isna().sum().to_string())

# Use readable scaled predictors for stable interpretation:
# price effect per 10 currency units;
# support effect per additional ticket;
# sessions effect per 5 sessions;
# discount effect per 5 percentage points.
model_df["price_per_10"] = model_df["السعر الشهري"] / 10
model_df["sessions_per_5"] = model_df["متوسط الجلسات الشهرية"] / 5
model_df["discount_per_5"] = model_df["نسبة الخصم"] / 5

# ============================================================
# 5. ADJUSTED LOGISTIC REGRESSION
# ============================================================


formula = (
    '_churn ~ Q("تذاكر الدعم") + '
    'sessions_per_5 + discount_per_5 + '
    'C(Q("الباقة")) + C(Q("دورة الفوترة")) + '
    'C(Q("وسيلة الدفع")) + C(Q("حضر التهيئة"))'
)

print("\n========== 8. ADJUSTED LOGISTIC REGRESSION ==========")

try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.logit(formula=formula, data=model_df).fit(
            disp=False, maxiter=200
        )

    for warning in caught:
        print("Model warning:", str(warning.message))

    ci = model.conf_int()

    results = pd.DataFrame({
        "Coefficient": model.params,
        "Odds_Ratio": np.exp(model.params),
        "CI_Lower_95": np.exp(ci[0]),
        "CI_Upper_95": np.exp(ci[1]),
        "P_Value": model.pvalues
    })

    results = results.drop(index="Intercept", errors="ignore")
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.sort_values("P_Value")

    print("\nAdjusted odds ratios and 95% confidence intervals:")
    print(results.round(4).to_string())

    print("\nPseudo R-squared:", round(model.prsquared, 4))
    print("Converged:", model.mle_retvals.get("converged", "unknown"))

    results.to_csv(OUTPUT, encoding="utf-8-sig")
    print("\nSaved regression results:", OUTPUT)

except Exception as error:
    print("Interpretation model could not be fitted:", repr(error))
    results = None

# ============================================================
# 6. HOLDOUT EVALUATION FOR IMBALANCED DATA
# ============================================================

print("\n========== 9. HOLDOUT MODEL EVALUATION ==========")

X = model_df[PREDICTOR_NUMERIC + PREDICTOR_CATEGORICAL].copy()
y = model_df["_churn"].astype(int)

if y.nunique() != 2 or y.value_counts().min() < 2:
    print("Not enough cases in both classes for stratified evaluation.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="constant", fill_value="Missing"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, PREDICTOR_NUMERIC),
        ("categorical", categorical_pipeline, PREDICTOR_CATEGORICAL)
    ])

    # class_weight='balanced' gives the minority class more weight.
    # This does not guarantee good predictions; metrics must be checked.
    predictive_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])

    predictive_model.fit(X_train, y_train)

    probabilities = predictive_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    prevalence = float(y_test.mean())

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Churn cases in test set:", int(y_test.sum()))
    print("Churn prevalence in test set: {:.4f}".format(prevalence))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))

    if y_test.nunique() == 2:
        print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
        ap = average_precision_score(y_test, probabilities)
        print("Average Precision:", round(ap, 4))
        print("Average Precision baseline (test churn prevalence):",
              round(prevalence, 4))

    print(
        "\nNote: the 0.5 classification threshold is a default, "
        "not necessarily the best business threshold."
    )
    print(
        "Do not select a final threshold using the test set. "
        "Threshold selection should be done with training/validation data."
    )

# ============================================================
# 7. INTERPRETATION NOTES
# ============================================================

print("\n========== 10. INTERPRETATION NOTES ==========")
print("Odds ratios above 1 indicate higher odds; below 1 indicate lower odds.")
print("Categorical effects are relative to the reference category.")
print("Sessions odds ratio is per 5 additional monthly sessions.")
print("Discount odds ratio is per 5 percentage points.")
print("Support tickets odds ratio is per additional ticket.")
print("Months retained was excluded because its timing may cause leakage.")
print("Associations do not prove causation.")
print("Predictive performance and explanatory associations are different.")

print("\n========== ANALYSIS FINISHED ==========")
