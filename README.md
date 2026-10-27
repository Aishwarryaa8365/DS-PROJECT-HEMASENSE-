# HemaSense – Smart Hematology Diagnostic Assistant

An AI-powered web application that analyses Complete Blood Count (CBC) parameters
and predicts hematological disorders using machine learning.

## Features

- Blood disease prediction from 14 CBC parameters (Random Forest model)
- Confidence score and risk level for each prediction
- Per-diagnosis clinical recommendations, suggested tests, and lifestyle advice
- Admin dashboard with user stats, prediction history, and charts
- Customer dashboard with personal prediction history and profile
- Role-based access control (admin / customer)
- Secure password hashing with Werkzeug

## Folder Structure

```
HemaSense/
├── frontend/
│   ├── templates/          # Jinja2 HTML templates
│   └── static/
│       └── assets/
│           ├── css/        # style.css, dashboard.css, prediction.css
│           ├── js/         # charts.js, prediction.js, script.js
│           └── images/     # backgrounds, diseases, hero, icons
├── backend/
│   ├── app.py              # Flask application
│   ├── config.py           # Path & environment config
│   ├── requirements.txt    # Python dependencies
│   └── README.md
├── ml/
│   ├── saved_models/       # model.pkl, scaler.pkl, label_encoder.pkl, feature_names.pkl
│   ├── eda_results/        # EDA charts and summary
│   ├── diagnosed_cbc_data_v4.csv
│   ├── train_model.py
│   ├── train_5features.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── validate_model.py
│   ├── validate_decision_tree.py
│   └── README.md
├── database/
│   ├── .gitkeep
│   └── README.md           # DB initialisation instructions
├── run.py                  # Project entry point
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.9+
- pip

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/HemaSense.git
cd HemaSense

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r backend/requirements.txt
```

## Environment Configuration

```bash
# Copy the example file
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux

# Edit .env and set a strong secret key
SECRET_KEY=your-secret-key-here
```

## Database Initialisation

The SQLite database is created automatically on first run.
See `database/README.md` for details.

## Running the Application

```bash
python run.py
```

Open your browser at: **http://127.0.0.1:5000**

Default admin credentials: `admin` / `admin123`

## ML Model

The trained Random Forest model is stored in `ml/saved_models/`.
To retrain:

```bash
python ml/train_model.py
```

To validate:

```bash
python ml/validate_model.py
```

## GitHub Preparation

```bash
git init
git add .
git commit -m "Initial commit – HemaSense restructured"
git remote add origin https://github.com/<your-username>/HemaSense.git
git push -u origin main
```

The `.gitignore` excludes `*.db`, `.env`, `__pycache__/`, and virtual environments.
