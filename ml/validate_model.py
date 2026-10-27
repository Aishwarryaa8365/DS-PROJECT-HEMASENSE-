"""
validate_model.py
-----------------
5-fold Stratified Cross-Validation for the Decision Tree classifier.

USAGE:
    cd ml/
    python validate_model.py

Output: ml/saved_models/cv_report.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from preprocess import remove_impossible_values

DATASET_PATH = "diagnosed_cbc_data_v4.csv"
TARGET_COL   = "Diagnosis"
SAVE_DIR     = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load and clean (mirrors preprocess.py steps 1-3) ─────────────────────────
df = pd.read_csv(DATASET_PATH)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df, _ = remove_impossible_values(df)

le = LabelEncoder()
X  = df.drop(columns=[TARGET_COL]).values
y  = le.fit_transform(df[TARGET_COL].values)

# ── Train Decision Tree on the same 80/20 split used in train_model.py ────────
# (for depth/leaf inspection only — does NOT save or overwrite model.pkl)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr_s, y_tr)

depth    = dt.get_depth()
n_leaves = dt.get_n_leaves()

# ── 5-fold Stratified CV using Pipeline (prevents data leakage) ───────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dt",     DecisionTreeClassifier(random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy"  : "accuracy",
    "precision" : "precision_weighted",
    "recall"    : "recall_weighted",
    "f1"        : "f1_weighted",
}

cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                             return_train_score=False, n_jobs=-1)

metrics = {
    "Accuracy"  : cv_results["test_accuracy"],
    "Precision" : cv_results["test_precision"],
    "Recall"    : cv_results["test_recall"],
    "F1-Score"  : cv_results["test_f1"],
}

# ── Print results ─────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 62)
lines.append("  DECISION TREE — 5-FOLD STRATIFIED CROSS-VALIDATION")
lines.append("=" * 62)

lines.append(f"\n  Tree Depth       : {depth}")
lines.append(f"  Number of Leaves : {n_leaves}\n")

lines.append(f"  {'Metric':<12} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8}  {'Mean':>8} {'Std':>8}")
lines.append("  " + "-" * 72)

for metric, scores in metrics.items():
    folds = "  ".join(f"{s:>8.4f}" for s in scores)
    lines.append(f"  {metric:<12} {folds}  {scores.mean():>8.4f} {scores.std():>8.4f}")

lines.append("\n" + "=" * 62)

report = "\n".join(lines)
print(report)

# ── Save report ───────────────────────────────────────────────────────────────
report_path = os.path.join(SAVE_DIR, "cv_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n  Report saved → {report_path}")
