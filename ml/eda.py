"""
eda.py
------
Exploratory Data Analysis for the HemaSense CBC dataset.
Applies the same cleaning as preprocess.py, then generates and saves plots.

USAGE:
    cd ml/
    python eda.py

Output: ml/eda_results/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import remove_impossible_values

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "diagnosed_cbc_data_v4.csv"
TARGET_COL   = "Diagnosis"
OUT_DIR      = "eda_results"
FEATURES     = ["WBC","LYMp","NEUTp","LYMn","NEUTn",
                 "RBC","HGB","HCT","MCV","MCH","MCHC","PLT","PDW","PCT"]

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["axes.grid"] = True

# ── Load & clean (mirrors preprocess.py steps 1-3) ───────────────────────────
df = pd.read_csv(DATASET_PATH)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df, _ = remove_impossible_values(df)

# ── 1. Dataset shape ──────────────────────────────────────────────────────────
print(f"Shape after cleaning : {df.shape}")

# ── 2. Diagnosis class distribution ──────────────────────────────────────────
print("\nClass distribution:")
print(df[TARGET_COL].value_counts().to_string())

fig, ax = plt.subplots(figsize=(10, 5))
counts = df[TARGET_COL].value_counts()
bars = ax.barh(counts.index, counts.values, color="steelblue")
ax.set_title("Diagnosis Class Distribution")
ax.set_xlabel("Count")
ax.set_ylabel("")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close(fig)
print("\nSaved: class_distribution.png")

# ── 3. Summary statistics ─────────────────────────────────────────────────────
stats = df[FEATURES].describe().T
print("\nSummary statistics:")
print(stats.to_string())
stats.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"))
print("Saved: summary_statistics.csv")

# ── 4. Histograms ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("CBC Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "histograms.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: histograms.png")

# ── 5. Boxplots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="lightcoral"))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("")
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("CBC Feature Boxplots", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "boxplots.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: boxplots.png")

# ── 6. Correlation heatmap ────────────────────────────────────────────────────
corr = df[FEATURES].corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(FEATURES, fontsize=9)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="black")
ax.set_title("CBC Feature Correlation Heatmap")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved: correlation_heatmap.png")

print(f"\nAll EDA outputs saved to: {OUT_DIR}/")
