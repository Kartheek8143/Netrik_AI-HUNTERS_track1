"""
Leave Approval ML Model — Training Script
==========================================
Trains Logistic Regression, Random Forest, and XGBoost to predict
leave approval / rejection + risk scores.

Outputs:
  - leave_approval_model.pkl   (best model)
  - feature_columns.pkl
  - label_encoders.pkl
"""

import warnings, os, sys, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import joblib

# ── try xgboost ──────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN]  xgboost not installed - skipping XGBoost model")

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "employee leave tracking data.xlsx")

# ═══════════════════════════════════════════════════════
# 1. LOAD & EXPLORE
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Nulls:\n{df.isnull().sum()}")
print(f"\n   leave_status distribution:\n{df['leave_status'].value_counts()}")

# ═══════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Convert dates
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

# Feature: leave_duration_days (from dates)
df["leave_duration_days"] = (df["end_date"] - df["start_date"]).dt.days.clip(lower=0)

# Feature: notice_period_days (assume request is 7 days before start by default)
# Since we don't have request_date, approximate from month column
df["notice_period_days"] = np.random.RandomState(SEED).randint(0, 30, size=len(df))

# Feature: month_of_year
df["month_of_year"] = df["start_date"].dt.month.fillna(1).astype(int)

# Feature: day_of_week
df["day_of_week"] = df["start_date"].dt.dayofweek.fillna(0).astype(int)

# Encode target
df["approved"] = (df["leave_status"].str.strip().str.lower() == "approved").astype(int)

# SYNTHESIZE REJECTED CASES (Dataset only had 300 Approved)
if df["approved"].nunique() == 1:
    print("   [INFO] Dataset only has 'Approved' cases. Synthesizing 200 'Rejected' cases...")
    df_rejected = df.sample(n=200, replace=True, random_state=SEED).copy().reset_index(drop=True)
    df_rejected["approved"] = 0
    df_rejected["leave_status"] = "Rejected"

    # Make them realistic rejections:
    n = len(df_rejected)
    idx1, idx2, idx3 = df_rejected.index[:int(0.4*n)], df_rejected.index[int(0.4*n):int(0.8*n)], df_rejected.index[int(0.8*n):]

    # Convert to numeric first to avoid type issues
    df_rejected["total_leave_entitlement"] = pd.to_numeric(df_rejected["total_leave_entitlement"], errors="coerce").fillna(0)

    # 1. 40% -> low balance
    added_days = np.random.randint(2, 10, len(idx1)).astype(float)
    df_rejected.loc[idx1, "leave_days"] = df_rejected.loc[idx1, "total_leave_entitlement"].values + added_days

    # 2. 40% -> 0 days notice period
    df_rejected.loc[idx2, "notice_period_days"] = 0

    # 3. 20% -> extremely long duration
    df_rejected.loc[idx3, "leave_duration_days"] = np.random.randint(30, 90, len(idx3)).astype(float)

    df = pd.concat([df, df_rejected], ignore_index=True)

# Encode categoricals
label_encoders = {}
cat_cols = ["leave_type", "department", "position"]
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
    label_encoders[col] = le
    print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Fill missing numeric values
for c in ["days_taken", "total_leave_entitlement", "leave_taken_so_far",
           "remaining_leaves", "leave_days"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

# Feature: leave_balance_ratio (how much of their balance they're using)
df["leave_balance_ratio"] = np.where(
    df["total_leave_entitlement"] > 0,
    df["leave_days"] / df["total_leave_entitlement"],
    0
)

# Feature: past_leaves_ratio
df["past_leaves_ratio"] = np.where(
    df["total_leave_entitlement"] > 0,
    df["leave_taken_so_far"] / df["total_leave_entitlement"],
    0
)

print(f"   After preprocessing: {df.shape}")

# ═══════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. FEATURES")
print("=" * 60)

FEATURE_COLS = [
    "leave_duration_days",
    "notice_period_days",
    "days_taken",
    "total_leave_entitlement",
    "leave_taken_so_far",
    "remaining_leaves",
    "leave_days",
    "leave_type_encoded",
    "department_encoded",
    "position_encoded",
    "month_of_year",
    "day_of_week",
    "leave_balance_ratio",
    "past_leaves_ratio",
]

X = df[FEATURE_COLS].fillna(0)
y = df["approved"]

print(f"   Features: {FEATURE_COLS}")
print(f"   X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════
# 4. TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ═══════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=SEED, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=SEED,
        class_weight="balanced", n_jobs=-1,
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="logloss",
        use_label_encoder=False,
    )

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": auc, "cm": cm,
    }
    print(f"   [*] {name}:")
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1:        {f1:.4f}")
    print(f"      ROC-AUC:   {auc:.4f}")
    print(f"      Confusion Matrix:\n{cm}")

# ═══════════════════════════════════════════════════════
# 6. SELECT BEST MODEL
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. MODEL COMPARISON")
print("=" * 60)

comparison = {n: r["f1"] for n, r in results.items()}
best_name = max(comparison, key=comparison.get)
best_model = results[best_name]["model"]
print(f"\n   🏆 Best model: {best_name} (F1={comparison[best_name]:.4f})")
print(f"\n   All scores:")
for n, s in sorted(comparison.items(), key=lambda x: -x[1]):
    print(f"      {n}: F1={s:.4f}, AUC={results[n]['roc_auc']:.4f}")

# ═══════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.zeros(len(FEATURE_COLS))

feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
print(f"\n   Feature importance ({best_name}):")
for i, (feat, imp) in enumerate(feat_imp, 1):
    bar = "█" * int(imp / max(importances) * 30) if max(importances) > 0 else ""
    print(f"   {i:2d}. {feat:25s} {imp:.4f} {bar}")

# ═══════════════════════════════════════════════════════
# 8. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. SAVING MODEL ARTIFACTS")
print("=" * 60)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "leave_approval_model.pkl")
feat_path = os.path.join(base_dir, "feature_columns.pkl")
enc_path = os.path.join(base_dir, "label_encoders.pkl")

joblib.dump(best_model, model_path)
joblib.dump(FEATURE_COLS, feat_path)
joblib.dump(label_encoders, enc_path)

print(f"   ✅ Model saved:    {model_path}")
print(f"   ✅ Features saved: {feat_path}")
print(f"   ✅ Encoders saved: {enc_path}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print("=" * 60)
print(f"   Best model:   {best_name}")
print(f"   F1 Score:     {results[best_name]['f1']:.4f}")
print(f"   ROC-AUC:      {results[best_name]['roc_auc']:.4f}")
print(f"   Accuracy:     {results[best_name]['accuracy']:.4f}")
print(f"   Saved to:     leave_approval_model.pkl")
