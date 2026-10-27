"""
train_5features.py
------------------
Trains and evaluates a Decision Tree using ONLY 5 CBC features:
    WBC, HGB, RBC, MCV, PLT

Compares result against the existing 14-feature Decision Tree.
Does NOT overwrite any existing saved model.

USAGE:
    cd ml/
    python train_5features.py

Saves to ml/saved_models/:
    model_5features.pkl
    scaler_5features.pkl
    label_encoder_5features.pkl
    feature_names_5features.pkl
    cm_5features.png
    comparison_report.txt
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import remove_impossible_values

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "diagnosed_cbc_data_v4.csv"
TARGET_COL     = "Diagnosis"
SAVE_DIR       = "saved_models"
FEATURES_5     = ["WBC", "HGB", "RBC", "MCV", "PLT"]
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load and clean (same rules as preprocess.py steps 1-3) ───────────────────
print("=" * 62)
print("  LOADING AND CLEANING DATASET")
print("=" * 62)
df = pd.read_csv(DATASET_PATH)
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df, n_removed = remove_impossible_values(df)
print(f"  Rows loaded       : {before}")
print(f"  Duplicates removed: {before - len(df) - n_removed}")
print(f"  Outliers removed  : {n_removed}")
print(f"  Rows after clean  : {len(df)}")

# ── Verify all 5 features exist ───────────────────────────────────────────────
missing_cols = [f for f in FEATURES_5 if f not in df.columns]
if missing_cols:
    print(f"\nERROR: These features are not in the dataset: {missing_cols}")
    sys.exit(1)

# ── Encode target ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL].values)
class_names = list(le.classes_)

print(f"\n  Target classes ({len(class_names)}):")
for i, c in enumerate(class_names):
    count = (df[TARGET_COL] == c).sum()
    print(f"    [{i}] {c:<40} n={count}")

# ── Feature analysis: per-class means for the 5 features ─────────────────────
print("\n" + "=" * 62)
print("  FEATURE ANALYSIS — Per-class means for 5 proposed features")
print("=" * 62)
analysis = df.groupby(TARGET_COL)[FEATURES_5].mean().round(2)
print(analysis.to_string())

# ── Stratified 80/20 split ────────────────────────────────────────────────────
X5 = df[FEATURES_5].values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X5, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler5 = StandardScaler()
X_train = scaler5.fit_transform(X_train_raw)
X_test  = scaler5.transform(X_test_raw)

# ── Train 5-feature Decision Tree ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  TRAINING 5-FEATURE DECISION TREE")
print("=" * 62)
dt5 = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt5.fit(X_train, y_train)
print(f"  Tree depth   : {dt5.get_depth()}")
print(f"  Leaf nodes   : {dt5.get_n_leaves()}")

# ── Held-out test evaluation ──────────────────────────────────────────────────
y_pred = dt5.predict(X_test)
acc5   = accuracy_score(y_test, y_pred)
prec5  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec5   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f15    = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\n  Test Accuracy  : {acc5:.4f}")
print(f"  Test Precision : {prec5:.4f}")
print(f"  Test Recall    : {rec5:.4f}")
print(f"  Test F1-Score  : {f15:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
thresh = cm.max() / 2
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=8, color="white" if cm[i, j] > thresh else "black")
ax.set_title("Confusion Matrix — 5-Feature Decision Tree")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "cm_5features.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"\n  Confusion matrix saved → {cm_path}")

# ── Feature importance ────────────────────────────────────────────────────────
print("\n  Feature Importances:")
for feat, imp in sorted(zip(FEATURES_5, dt5.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"    {feat:<6} {imp:.4f}  {bar}")

# ── 5-fold stratified cross-validation ───────────────────────────────────────
print("\n" + "=" * 62)
print("  5-FOLD STRATIFIED CROSS-VALIDATION (5 features)")
print("=" * 62)
from sklearn.pipeline import Pipeline
pipeline5 = Pipeline([
    ("scaler", StandardScaler()),
    ("dt",     DecisionTreeClassifier(random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy"  : "accuracy",
    "precision" : "precision_weighted",
    "recall"    : "recall_weighted",
    "f1"        : "f1_weighted",
}
cv5 = cross_validate(pipeline5, X5, y, cv=cv, scoring=scoring,
                     return_train_score=True, n_jobs=-1)

cv5_metrics = {
    "Accuracy"  : cv5["test_accuracy"],
    "Precision" : cv5["test_precision"],
    "Recall"    : cv5["test_recall"],
    "F1-Score"  : cv5["test_f1"],
}
print(f"\n  {'Metric':<12} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8}  {'Mean':>8} {'Std':>8}")
print("  " + "-" * 72)
for metric, scores in cv5_metrics.items():
    folds = "  ".join(f"{s:>8.4f}" for s in scores)
    print(f"  {metric:<12} {folds}  {scores.mean():>8.4f} {scores.std():>8.4f}")

cv5_train_acc = cv5["train_accuracy"]
print(f"\n  CV Train Accuracy : {cv5_train_acc.mean():.4f} ± {cv5_train_acc.std():.4f}")
print(f"  CV Train-Test Gap : {cv5_train_acc.mean() - cv5_metrics['Accuracy'].mean():.4f}")

# ── Load existing 14-feature results for comparison ───────────────────────────
print("\n" + "=" * 62)
print("  LOADING EXISTING 14-FEATURE MODEL FOR COMPARISON")
print("=" * 62)
model14       = joblib.load(os.path.join(SAVE_DIR, "model.pkl"))
scaler14      = joblib.load(os.path.join(SAVE_DIR, "scaler.pkl"))
le14          = joblib.load(os.path.join(SAVE_DIR, "label_encoder.pkl"))
feat14        = joblib.load(os.path.join(SAVE_DIR, "feature_names.pkl"))

# Reproduce the same 80/20 split on all 14 features
X14 = df[feat14].values
X14_tr_raw, X14_te_raw, y14_tr, y14_te = train_test_split(
    X14, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X14_te_scaled = scaler14.transform(X14_te_raw)
y14_pred = model14.predict(X14_te_scaled)

acc14  = accuracy_score(y14_te, y14_pred)
prec14 = precision_score(y14_te, y14_pred, average="weighted", zero_division=0)
rec14  = recall_score(y14_te, y14_pred, average="weighted", zero_division=0)
f114   = f1_score(y14_te, y14_pred, average="weighted", zero_division=0)

# CV for 14-feature model
pipeline14 = Pipeline([
    ("scaler", StandardScaler()),
    ("dt",     DecisionTreeClassifier(random_state=RANDOM_STATE)),
])
cv14 = cross_validate(pipeline14, X14, y, cv=cv, scoring=scoring,
                      return_train_score=False, n_jobs=-1)
cv14_acc_mean = cv14["test_accuracy"].mean()
cv14_acc_std  = cv14["test_accuracy"].std()
cv14_f1_mean  = cv14["test_f1"].mean()
cv14_f1_std   = cv14["test_f1"].std()

# ── Final comparison table ────────────────────────────────────────────────────
lines = []
lines.append("\n" + "=" * 70)
lines.append("  FINAL COMPARISON: 14-Feature Model vs 5-Feature Model")
lines.append("=" * 70)
lines.append(f"\n  {'Metric':<28} {'14 Features':>14} {'5 Features':>14} {'Difference':>12}")
lines.append("  " + "-" * 68)

rows = [
    ("Test Accuracy",        acc14,          acc5),
    ("Test Precision",       prec14,         prec5),
    ("Test Recall",          rec14,          rec5),
    ("Test F1-Score",        f114,           f15),
    ("CV Accuracy Mean",     cv14_acc_mean,  cv5_metrics["Accuracy"].mean()),
    ("CV Accuracy Std",      cv14_acc_std,   cv5_metrics["Accuracy"].std()),
    ("CV F1 Mean",           cv14_f1_mean,   cv5_metrics["F1-Score"].mean()),
    ("CV F1 Std",            cv14_f1_std,    cv5_metrics["F1-Score"].std()),
]
for label, v14, v5 in rows:
    diff = v5 - v14
    sign = "+" if diff >= 0 else ""
    lines.append(f"  {label:<28} {v14:>14.4f} {v5:>14.4f} {sign}{diff:>11.4f}")

lines.append("\n  Tree Structure:")
lines.append(f"    14-feature depth  : {model14.get_depth()}   leaves: {model14.get_n_leaves()}")
lines.append(f"    5-feature  depth  : {dt5.get_depth()}   leaves: {dt5.get_n_leaves()}")

lines.append("\n  Feature Importances (5-feature model):")
for feat, imp in sorted(zip(FEATURES_5, dt5.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    lines.append(f"    {feat:<6}  {imp:.4f}  ({'█' * int(imp * 30)})")

# Verdict
acc_drop = acc14 - acc5
f1_drop  = f114  - f15
lines.append("\n" + "=" * 70)
lines.append("  VERDICT")
lines.append("=" * 70)
if acc_drop <= 0.02 and f1_drop <= 0.02:
    lines.append(f"\n  The 5-feature model is SUITABLE for production use.")
    lines.append(f"  Accuracy drop : {acc_drop*100:+.2f}%  (within 2% threshold)")
    lines.append(f"  F1 drop       : {f1_drop*100:+.2f}%  (within 2% threshold)")
    lines.append(f"  Recommendation: SAFE to replace 14-feature model with 5-feature model.")
elif acc_drop <= 0.05 and f1_drop <= 0.05:
    lines.append(f"\n  The 5-feature model shows ACCEPTABLE performance loss.")
    lines.append(f"  Accuracy drop : {acc_drop*100:+.2f}%  (within 5% threshold)")
    lines.append(f"  F1 drop       : {f1_drop*100:+.2f}%  (within 5% threshold)")
    lines.append(f"  Recommendation: ACCEPTABLE trade-off for usability improvement.")
else:
    lines.append(f"\n  The 5-feature model shows SIGNIFICANT performance loss.")
    lines.append(f"  Accuracy drop : {acc_drop*100:+.2f}%")
    lines.append(f"  F1 drop       : {f1_drop*100:+.2f}%")
    lines.append(f"  Recommendation: Consider keeping 14-feature model or selecting different features.")
lines.append("=" * 70)

report = "\n".join(lines)
print(report)

# ── Save report ───────────────────────────────────────────────────────────────
report_path = os.path.join(SAVE_DIR, "comparison_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n  Report saved → {report_path}")

# ── Save 5-feature artifacts ──────────────────────────────────────────────────
joblib.dump(dt5,            os.path.join(SAVE_DIR, "model_5features.pkl"))
joblib.dump(scaler5,        os.path.join(SAVE_DIR, "scaler_5features.pkl"))
joblib.dump(le,             os.path.join(SAVE_DIR, "label_encoder_5features.pkl"))
joblib.dump(FEATURES_5,     os.path.join(SAVE_DIR, "feature_names_5features.pkl"))

print(f"\n  Saved → {SAVE_DIR}/model_5features.pkl")
print(f"  Saved → {SAVE_DIR}/scaler_5features.pkl")
print(f"  Saved → {SAVE_DIR}/label_encoder_5features.pkl")
print(f"  Saved → {SAVE_DIR}/feature_names_5features.pkl")
print("\nDone. Existing model.pkl is unchanged.")
