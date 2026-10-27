"""
train_model.py
--------------
Trains and compares 4 classifiers for HemaSense CBC disease prediction.

USAGE:
    cd ml/
    python train_model.py

Saves best model artifacts to ml/saved_models/
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import load_and_preprocess

DATASET_PATH  = "diagnosed_cbc_data_v4.csv"
TARGET_COLUMN = "Diagnosis"
SAVE_DIR      = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_confusion_matrix(cm, class_names, model_name):
    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = os.path.join(SAVE_DIR, f"cm_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {fname}")


def evaluate(name, model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  Model     : {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"{'─'*55}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, name)

    return {"name": name, "model": model,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_all_models():
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = \
        load_and_preprocess(filepath=DATASET_PATH, target_column=TARGET_COLUMN)

    class_names = list(label_encoder.classes_)

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test, class_names))

    # ── Comparison table (sorted by F1) ──────────────────────────────────────
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)

    print("\n" + "═"*65)
    print("  MODEL COMPARISON SUMMARY  (sorted by F1-Score)")
    print("═"*65)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "─"*60)
    for r in results_sorted:
        print(f"  {r['name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f}"
              f" {r['recall']:>10.4f} {r['f1']:>10.4f}")
    print("═"*65)

    # ── Best model by F1 ─────────────────────────────────────────────────────
    best = results_sorted[0]
    print(f"\nBest model : {best['name']}")
    print(f"  Accuracy  : {best['accuracy']:.4f}")
    print(f"  Precision : {best['precision']:.4f}")
    print(f"  Recall    : {best['recall']:.4f}")
    print(f"  F1-Score  : {best['f1']:.4f}")

    joblib.dump(best["model"],   os.path.join(SAVE_DIR, "model.pkl"))
    joblib.dump(scaler,          os.path.join(SAVE_DIR, "scaler.pkl"))
    joblib.dump(label_encoder,   os.path.join(SAVE_DIR, "label_encoder.pkl"))
    joblib.dump(feature_names,   os.path.join(SAVE_DIR, "feature_names.pkl"))

    print(f"\nSaved → {SAVE_DIR}/model.pkl")
    print(f"Saved → {SAVE_DIR}/scaler.pkl")
    print(f"Saved → {SAVE_DIR}/label_encoder.pkl")
    print(f"Saved → {SAVE_DIR}/feature_names.pkl")
    print("\nTraining complete.")


if __name__ == "__main__":
    train_all_models()
