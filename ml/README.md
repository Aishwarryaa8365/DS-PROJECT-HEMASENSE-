# HemaSense – ML Module

This folder contains the machine learning pipeline for the HemaSense hematology diagnostic assistant.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Data loading, cleaning, encoding, scaling, and train/test split |
| `train_model.py` | Trains and compares 4 classifiers, saves the best model |
| `dataset.csv` | ← Place your CBC dataset here (not included in repo) |
| `saved_models/` | Auto-created after training; stores `.pkl` model files |

## Steps to Run

### 1. Add your dataset
Place your CSV file inside this `ml/` folder and name it `dataset.csv`  
(or update `DATASET_PATH` in `train_model.py`).

### 2. Set your target column
Open `train_model.py` and update:
```python
TARGET_COLUMN = "Diagnosis"   # replace with your actual column name
```
Also update the same in `preprocess.py` if running it standalone.

### 3. Run preprocessing (optional check)
```bash
cd ml
python preprocess.py
```

### 4. Train all models
```bash
cd ml
python train_model.py
```

This will:
- Print accuracy, precision, recall, F1-score for each model
- Save confusion matrix plots as `.png` files
- Save the best model to `saved_models/model.pkl`

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

## Output Files (after training)
```
ml/
├── saved_models/
│   ├── model.pkl           ← best trained model
│   ├── scaler.pkl          ← fitted StandardScaler
│   └── label_encoder.pkl   ← fitted LabelEncoder for target classes
└── confusion_*.png         ← confusion matrix plots
```

## Next Step
After training, the saved model will be integrated into `app.py` to power the `/prediction` route.
