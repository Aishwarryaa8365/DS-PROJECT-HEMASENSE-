"""
validate_decision_tree.py
--------------------------
Validates the Decision Tree classifier for the HemaSense CBC dataset.

Checks:
  1. Overfitting  — train accuracy vs test accuracy
  2. 5-fold stratified cross-validation (Accuracy, Precision, Recall, F1)
  3. Tree structure  — max depth and number of leaves
  4. CV vs held-out test comparison

Uses a Pipeline (StandardScaler → DecisionTree) inside CV folds
to prevent data leakage.

USAGE:
    cd ml/
    python validate_decision_tree.py

Output: ml/saved_models/dt_validation_report.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score
)

from preprocess import remove_impossible_values

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "diagnosed_cbc_data_v4.csv"
TARGET_COL   = "Diagnosis"
SAVE_DIR     = "saved_models"
REPORT_PATH  = os.path.join(SAVE_DIR, "dt_validation_report.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Reproduce the exact same held-out split used in train_model.py ────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Load & clean (same steps as preprocess.py 1-3, without scaling/splitting) ─
df = pd.read_csv(DATASET_PATH)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df, _ = remove_impossible_values(df)

X = df.drop(columns=[TARGET_COL]).values
le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL].values)
class_names = list(le.classes_)

# ── Reproduce the exact held-out split ───────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Scale using only train statistics (mirrors preprocess.py)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# ── Train Decision Tree on the held-out split ─────────────────────────────────
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)

# ── 1. Overfitting check ──────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc  = accuracy_score(y_test,  dt.predict(X_test))
test_prec = precision_score(y_test, dt.predict(X_test), average="weighted", zero_division=0)
test_rec  = recall_score(y_test,    dt.predict(X_test), average="weighted", zero_division=0)
test_f1   = f1_score(y_test,        dt.predict(X_test), average="weighted", zero_division=0)
gap       = train_acc - test_acc

# ── 2. Tree structure ─────────────────────────────────────────────────────────
depth       = dt.get_depth()
n_leaves    = dt.get_n_leaves()
n_features  = X_train.shape[1]
n_train     = X_train.shape[0]

# ── 3. 5-fold stratified cross-validation (Pipeline prevents leakage) ─────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dt",     DecisionTreeClassifier(random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy"  : "accuracy",
    "precision" : "precision_weighted",
    "recall"    : "recall_weighted",
    "f1"        : "f1_weighted",
}

cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                             return_train_score=True, n_jobs=-1)

cv_metrics = {
    "accuracy"  : cv_results["test_accuracy"],
    "precision" : cv_results["test_precision"],
    "recall"    : cv_results["test_recall"],
    "f1"        : cv_results["test_f1"],
}
cv_train_acc = cv_results["train_accuracy"]

# ── Build report ──────────────────────────────────────────────────────────────
lines = []

def h(title):
    lines.append("\n" + "═"*60)
    lines.append(f"  {title}")
    lines.append("═"*60)

def row(label, val):
    lines.append(f"  {label:<30} {val}")

h("1. TREE STRUCTURE")
row("Max depth",          str(depth))
row("Number of leaves",   str(n_leaves))
row("Number of features", str(n_features))
row("Training samples",   str(n_train))

h("2. OVERFITTING CHECK")
row("Train accuracy",     f"{train_acc:.4f}")
row("Test  accuracy",     f"{test_acc:.4f}")
row("Gap (train - test)", f"{gap:.4f}")
if gap > 0.05:
    verdict = "⚠  LIKELY OVERFITTING  (gap > 0.05)"
elif gap > 0.02:
    verdict = "⚠  MILD OVERFITTING    (gap 0.02–0.05)"
else:
    verdict = "✓  NO SIGNIFICANT OVERFITTING  (gap ≤ 0.02)"
row("Verdict", verdict)

h("3. HELD-OUT TEST PERFORMANCE")
row("Accuracy",  f"{test_acc:.4f}")
row("Precision", f"{test_prec:.4f}")
row("Recall",    f"{test_rec:.4f}")
row("F1-Score",  f"{test_f1:.4f}")

h("4. 5-FOLD STRATIFIED CROSS-VALIDATION")
lines.append(f"\n  {'Metric':<12} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8}  {'Mean':>8} {'Std':>8}")
lines.append("  " + "─"*72)
for metric, scores in cv_metrics.items():
    fold_vals = "  ".join(f"{s:>8.4f}" for s in scores)
    lines.append(f"  {metric:<12} {fold_vals}  {scores.mean():>8.4f} {scores.std():>8.4f}")

cv_train_line = "  ".join(f"{s:>8.4f}" for s in cv_train_acc)
lines.append(f"\n  {'CV Train Acc':<12} {cv_train_line}  {cv_train_acc.mean():>8.4f} {cv_train_acc.std():>8.4f}")
cv_gap = cv_train_acc.mean() - cv_metrics["accuracy"].mean()
lines.append(f"\n  CV Train-Test Accuracy Gap : {cv_gap:.4f}")

h("5. CV vs HELD-OUT TEST COMPARISON")
lines.append(f"\n  {'Metric':<12} {'CV Mean':>10} {'CV Std':>10} {'Test':>10} {'Diff':>10}")
lines.append("  " + "─"*52)
for metric, scores in cv_metrics.items():
    test_val = {"accuracy": test_acc, "precision": test_prec,
                "recall": test_rec, "f1": test_f1}[metric]
    diff = test_val - scores.mean()
    lines.append(f"  {metric:<12} {scores.mean():>10.4f} {scores.std():>10.4f}"
                 f" {test_val:>10.4f} {diff:>+10.4f}")

h("6. RELIABILITY VERDICT")
cv_f1_mean = cv_metrics["f1"].mean()
cv_f1_std  = cv_metrics["f1"].std()
cv_acc_mean = cv_metrics["accuracy"].mean()

lines.append(f"\n  Held-out test F1   : {test_f1:.4f}")
lines.append(f"  CV mean F1         : {cv_f1_mean:.4f}  ±  {cv_f1_std:.4f}")
lines.append(f"  CV mean accuracy   : {cv_acc_mean:.4f}")
lines.append(f"  Overfitting gap    : {gap:.4f}  (train={train_acc:.4f}, test={test_acc:.4f})")
lines.append(f"  Tree depth         : {depth}  |  Leaves : {n_leaves}")
lines.append("")

# Reliability logic
reliable = True
reasons  = []
warnings = []

if gap > 0.05:
    reliable = False
    reasons.append(f"- Train-test gap is {gap:.4f} (> 0.05) — tree memorised training data.")
elif gap > 0.02:
    warnings.append(f"- Mild train-test gap ({gap:.4f}). Monitor on new data.")

if cv_f1_std > 0.05:
    reliable = False
    reasons.append(f"- CV F1 std is {cv_f1_std:.4f} (> 0.05) — unstable across folds.")
elif cv_f1_std > 0.02:
    warnings.append(f"- CV F1 std is {cv_f1_std:.4f}. Slight fold-to-fold variation.")

if abs(test_f1 - cv_f1_mean) > 0.05:
    reliable = False
    reasons.append(f"- Test F1 ({test_f1:.4f}) deviates from CV mean ({cv_f1_mean:.4f}) by > 0.05.")

if depth > 30:
    warnings.append(f"- Tree depth {depth} is very large. Consider pruning for production.")

if reliable:
    lines.append("  CONCLUSION: The 99.59% test performance APPEARS RELIABLE.")
    lines.append("  CV results are consistent with the held-out test score.")
    if warnings:
        lines.append("\n  Warnings (non-critical):")
        lines.extend(f"  {w}" for w in warnings)
else:
    lines.append("  CONCLUSION: The 99.59% test performance IS NOT FULLY RELIABLE.")
    lines.append("  Reasons:")
    lines.extend(f"  {r}" for r in reasons)
    if warnings:
        lines.append("\n  Additional warnings:")
        lines.extend(f"  {w}" for w in warnings)

lines.append("\n" + "═"*60)

report = "\n".join(lines)

# ── Print to console ──────────────────────────────────────────────────────────
print(report)

# ── Save to file ──────────────────────────────────────────────────────────────
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nReport saved → {REPORT_PATH}")
