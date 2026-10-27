"""
preprocess.py
-------------
Preprocessing pipeline for the HemaSense CBC disease classification dataset.
Dataset: diagnosed_cbc_data_v4.csv

Cleaning rules applied (approved):
  1. Remove exact duplicate rows.
  2. Remove rows with physically impossible CBC values (negatives, zeros where invalid).
  3. Remove rows with clearly erroneous extreme outliers per column.
  4. Rows with repeated LYMp/NEUTp/LYMn/NEUTn values are KEPT.
  5. Diagnosis column is kept unchanged as the target.

USAGE:
    from preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = \
        load_and_preprocess(
            filepath="diagnosed_cbc_data_v4.csv",
            target_column="Diagnosis"
        )
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Clinical thresholds for outlier removal ───────────────────────────────────
# Values outside these ranges are physically impossible or clearly erroneous.
# Ranges are intentionally wide to only remove obvious data entry errors.
CBC_BOUNDS = {
    # column  : (min_valid, max_valid)
    "WBC"   : (0.1,  100.0),   # ×10³/µL  — WBC > 100 is extreme leukemia territory
    "LYMp"  : (0.0,  100.0),   # %
    "NEUTp" : (0.0,  100.0),   # %
    "LYMn"  : (0.0,   50.0),   # ×10³/µL
    "NEUTn" : (0.0,   80.0),   # ×10³/µL
    "RBC"   : (0.5,   10.0),   # ×10⁶/µL
    "HGB"   : (1.0,   25.0),   # g/dL      — negative or >25 is impossible
    "HCT"   : (5.0,   70.0),   # %         — >70 or negative is impossible
    "MCV"   : (50.0, 150.0),   # fL        — >150 or <50 is erroneous
    "MCH"   : (10.0,  60.0),   # pg
    "MCHC"  : (20.0,  45.0),   # g/dL
    "PLT"   : (1.0,  800.0),   # ×10³/µL
    "PDW"   : (5.0,   30.0),   # %         — PDW=97 is impossible
    "PCT"   : (0.01,   1.0),   # %         — PCT=12 is impossible
}


def remove_impossible_values(df):
    """
    Drops rows where any CBC feature falls outside its valid clinical range.
    Returns the cleaned DataFrame and a count of rows removed.
    """
    before = len(df)
    for col, (lo, hi) in CBC_BOUNDS.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
    removed = before - len(df)
    return df.reset_index(drop=True), removed


def load_and_preprocess(filepath, target_column="Diagnosis",
                        test_size=0.2, random_state=42):
    """
    Loads, cleans, and preprocesses the CBC dataset.

    Parameters
    ----------
    filepath      : str   – path to the CSV file
    target_column : str   – name of the diagnosis/label column
    test_size     : float – fraction reserved for testing (default 0.2)
    random_state  : int   – reproducibility seed

    Returns
    -------
    X_train, X_test  : scaled feature arrays (numpy)
    y_train, y_test  : encoded label arrays (numpy)
    feature_names    : list of feature column names
    label_encoder    : fitted LabelEncoder for the target
    scaler           : fitted StandardScaler for the features
    """

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[1] Loaded          : {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Drop exact duplicates ──────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[2] Duplicates removed : {before - len(df)}  →  {len(df)} rows remain")

    # ── 3. Remove physically impossible / erroneous CBC values ───────────────
    df, n_removed = remove_impossible_values(df)
    print(f"[3] Outlier rows removed : {n_removed}  →  {len(df)} rows remain")

    # ── 4. Verify target column ───────────────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── 5. Separate features and target ──────────────────────────────────────
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    feature_names = X.columns.tolist()
    print(f"[4] Features ({len(feature_names)})  : {feature_names}")
    print(f"[5] Target classes  : {sorted(y.unique().tolist())}")

    # ── 6. Confirm all features are numeric (no encoding needed for this dataset)
    non_numeric = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric:
        print(f"[6] Encoding categorical features: {non_numeric}")
        for col in non_numeric:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
    else:
        print(f"[6] All features are numeric — no encoding needed")

    # ── 7. Handle any remaining NaN (safety net) ─────────────────────────────
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"[7] Filling {nan_count} remaining NaN values with column median")
        X.fillna(X.median(), inplace=True)
    else:
        print(f"[7] No NaN values found")

    # ── 8. Encode target labels ───────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[8] Label encoding  : {dict(enumerate(label_encoder.classes_))}")

    # ── 9. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    print(f"[9] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── 10. Feature scaling ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"[10] StandardScaler applied")

    print("\nPreprocessing complete.\n")
    return X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler


# ── Run standalone to verify cleaning ────────────────────────────────────────
if __name__ == "__main__":
    DATASET_PATH  = "diagnosed_cbc_data_v4.csv"
    TARGET_COLUMN = "Diagnosis"

    X_train, X_test, y_train, y_test, features, le, scaler = load_and_preprocess(
        filepath=DATASET_PATH,
        target_column=TARGET_COLUMN
    )

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"Classes       : {list(le.classes_)}")
