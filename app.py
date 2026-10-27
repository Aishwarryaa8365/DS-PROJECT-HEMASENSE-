import os
import sqlite3
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, g)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hemasense-dev-secret-2025")

# ── Database path ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "hemasense.db")

# ── Load ML model artifacts once at startup ───────────────────────────────────
MODEL_DIR     = os.path.join(BASE_DIR, "ml", "saved_models")
model         = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# ── Per-diagnosis clinical content ────────────────────────────────────────────
DIAGNOSIS_INFO = {
    "Healthy": {
        "risk_level":     "Low",
        "recommendation": "CBC values are within normal range. Maintain a balanced diet and schedule routine check-ups annually.",
        "suggested_tests": ["Annual CBC", "Lipid Panel", "Blood Glucose"],
        "lifestyle":      ["Maintain a balanced diet rich in iron and vitamins", "Exercise regularly", "Stay hydrated", "Avoid smoking and excessive alcohol"],
        "doctor_note":    "No immediate consultation required. Continue routine annual health screening.",
    },
    "Iron deficiency anemia": {
        "risk_level":     "High",
        "recommendation": "Iron deficiency detected. Begin iron supplementation as advised and monitor CBC every 4 weeks.",
        "suggested_tests": ["Serum Ferritin", "Serum Iron & TIBC", "Peripheral Blood Smear", "Reticulocyte Count"],
        "lifestyle":      ["Increase iron-rich foods (spinach, lentils, red meat)", "Avoid tea/coffee with meals", "Take Vitamin C to boost iron absorption", "Maintain regular sleep schedule"],
        "doctor_note":    "Schedule an appointment with a hematologist within 7 days. Bring this report and recent blood test records.",
    },
    "Normocytic hypochromic anemia": {
        "risk_level":     "Medium",
        "recommendation": "Hypochromic pattern with normal cell size detected. Investigate underlying cause including iron or chronic disease.",
        "suggested_tests": ["Serum Ferritin", "Serum Iron", "CRP / ESR", "Peripheral Blood Smear"],
        "lifestyle":      ["Eat iron and protein-rich foods", "Avoid prolonged fasting", "Monitor fatigue and pallor", "Follow prescribed supplementation"],
        "doctor_note":    "Consult a hematologist within 2 weeks for further workup to rule out chronic disease anemia.",
    },
    "Normocytic normochromic anemia": {
        "risk_level":     "Medium",
        "recommendation": "Anemia with normal cell size and color. Likely due to chronic disease, renal insufficiency, or early bone marrow issue.",
        "suggested_tests": ["Reticulocyte Count", "Renal Function Tests", "Bone Marrow Biopsy (if indicated)", "Erythropoietin Level"],
        "lifestyle":      ["Ensure adequate protein intake", "Manage underlying chronic conditions", "Avoid strenuous activity until evaluated", "Stay well hydrated"],
        "doctor_note":    "Refer to a hematologist or nephrologist within 2 weeks for comprehensive evaluation.",
    },
    "Leukemia": {
        "risk_level":     "High",
        "recommendation": "CBC pattern is consistent with leukemia. Immediate specialist referral is critical.",
        "suggested_tests": ["Bone Marrow Biopsy", "Flow Cytometry", "Cytogenetics / Karyotyping", "LDH & Uric Acid"],
        "lifestyle":      ["Avoid infections — maintain strict hygiene", "Do not self-medicate", "Rest and avoid physical strain", "Seek immediate specialist care"],
        "doctor_note":    "Urgent referral to an oncologist or hematologist required. Do not delay — present to a hospital immediately.",
    },
    "Leukemia with thrombocytopenia": {
        "risk_level":     "High",
        "recommendation": "Leukemia with low platelet count detected. High bleeding risk. Immediate hospitalisation may be required.",
        "suggested_tests": ["Bone Marrow Biopsy", "Coagulation Panel (PT/APTT)", "Flow Cytometry", "Platelet Function Assay"],
        "lifestyle":      ["Avoid any activity with bleeding risk", "Do not take NSAIDs or aspirin", "Strict infection precautions", "Immediate medical supervision required"],
        "doctor_note":    "Emergency referral to oncology/hematology. Patient should be hospitalised for monitoring and treatment initiation.",
    },
    "Thrombocytopenia": {
        "risk_level":     "Medium",
        "recommendation": "Low platelet count detected. Investigate cause — immune, drug-induced, or bone marrow related.",
        "suggested_tests": ["Peripheral Blood Smear", "Coagulation Panel", "ANA / Anti-platelet Antibodies", "Bone Marrow Biopsy (if indicated)"],
        "lifestyle":      ["Avoid contact sports and injury risk", "Do not take aspirin or ibuprofen", "Report any unusual bruising or bleeding", "Follow prescribed treatment plan"],
        "doctor_note":    "Consult a hematologist within 1 week. Avoid medications that affect platelet function.",
    },
    "Other microcytic anemia": {
        "risk_level":     "Medium",
        "recommendation": "Small red blood cells detected. May indicate thalassemia, sideroblastic anemia, or lead poisoning.",
        "suggested_tests": ["Hemoglobin Electrophoresis", "Serum Lead Level", "Bone Marrow Iron Stain", "Peripheral Blood Smear"],
        "lifestyle":      ["Avoid iron supplementation without diagnosis confirmation", "Eat folate-rich foods", "Monitor symptoms of fatigue and pallor", "Follow specialist dietary advice"],
        "doctor_note":    "Refer to a hematologist for electrophoresis and further workup to identify the specific microcytic cause.",
    },
    "Macrocytic anemia": {
        "risk_level":     "Medium",
        "recommendation": "Large red blood cells detected. Likely B12 or folate deficiency. Neurological symptoms should be assessed.",
        "suggested_tests": ["Serum B12", "Serum Folate", "Peripheral Blood Smear", "Thyroid Function Tests"],
        "lifestyle":      ["Increase B12-rich foods (eggs, dairy, meat, fish)", "Take prescribed B12/folate supplements", "Limit alcohol consumption", "Report any numbness or tingling"],
        "doctor_note":    "Consult a physician within 2 weeks. B12 injections may be required if absorption is impaired.",
    },
}

DEFAULT_INFO = {
    "risk_level":     "Medium",
    "recommendation": "Abnormal CBC pattern detected. Please consult a hematologist for further evaluation.",
    "suggested_tests": ["Complete Blood Count (CBC)", "Peripheral Blood Smear", "Bone Marrow Biopsy (if indicated)"],
    "lifestyle":      ["Maintain a healthy diet", "Avoid self-medication", "Follow up with your physician"],
    "doctor_note":    "Consult a certified hematologist for a definitive diagnosis and treatment plan.",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Create tables if they don't exist and seed the admin account."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name     TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'customer',
                created_at    TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL,
                patient_name     TEXT,
                predicted_disease TEXT   NOT NULL,
                confidence       TEXT    NOT NULL,
                risk_level       TEXT    NOT NULL,
                created_at       TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # Seed admin if not present
        existing = conn.execute(
            "SELECT id FROM users WHERE username = 'admin'"
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO users (full_name, email, username, password_hash, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("Administrator", "admin@hemasense.ai", "admin",
                 generate_password_hash("admin123"), "admin",
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Access denied. Admins only.", "danger")
            return redirect(url_for("customer_dashboard"))
        return f(*args, **kwargs)
    return decorated

def customer_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "customer":
            flash("Access denied.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    # If already logged in, redirect to the correct dashboard
    if "user_id" in session:
        if session.get("role") == "admin":
            return redirect(url_for("dashboard"))
        return redirect(url_for("customer_dashboard"))

    error = None
    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password   = request.form.get("password", "").strip()

        print(f"[LOGIN] identifier='{identifier}' password_len={len(password)}")

        if not identifier or not password:
            error = "Please enter both username/email and password."
        else:
            db   = get_db()
            user = db.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (identifier, identifier)
            ).fetchone()

            if user is None:
                print(f"[LOGIN] No user found for identifier='{identifier}'")
                error = "Invalid username/email or password. Please try again."
            elif not check_password_hash(user["password_hash"], password):
                print(f"[LOGIN] Wrong password for user='{user['username']}'")
                error = "Invalid username/email or password. Please try again."
            else:
                print(f"[LOGIN] Success: user='{user['username']}' role='{user['role']}'")
                session.clear()
                session["user_id"]   = user["id"]
                session["username"]  = user["username"]
                session["full_name"] = user["full_name"]
                session["role"]      = user["role"]
                if user["role"] == "admin":
                    return redirect(url_for("dashboard"))
                return redirect(url_for("customer_dashboard"))

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    # Already logged in — send to the right dashboard
    if "user_id" in session:
        if session.get("role") == "admin":
            return redirect(url_for("dashboard"))
        return redirect(url_for("customer_dashboard"))

    error   = None
    success = None

    if request.method == "POST":
        full_name        = request.form.get("full_name", "").strip()
        email            = request.form.get("email", "").strip().lower()
        username         = request.form.get("username", "").strip().lower()
        password         = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not all([full_name, email, username, password, confirm_password]):
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        else:
            db = get_db()
            existing_email    = db.execute("SELECT id FROM users WHERE email = ?",    (email,)).fetchone()
            existing_username = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()

            if existing_email:
                error = "An account with this email already exists."
            elif existing_username:
                error = "This username is already taken. Please choose another."
            else:
                db.execute(
                    """INSERT INTO users (full_name, email, username, password_hash, role, created_at)
                       VALUES (?, ?, ?, ?, 'customer', ?)""",
                    (full_name, email, username,
                     generate_password_hash(password),
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                db.commit()
                flash("Account created successfully! Please log in.", "success")
                return redirect(url_for("login"))

    return render_template("signup.html", error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    response = redirect(url_for("login"))
    response.delete_cookie(app.config.get("SESSION_COOKIE_NAME", "session"))
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard")
@admin_required
def dashboard():
    db    = get_db()
    total_users       = db.execute("SELECT COUNT(*) FROM users WHERE role='customer'").fetchone()[0]
    total_predictions = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    high_risk         = db.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='High'").fetchone()[0]
    recent            = db.execute(
        """SELECT p.*, u.full_name FROM predictions p
           JOIN users u ON p.user_id = u.id
           ORDER BY p.created_at DESC LIMIT 10"""
    ).fetchall()
    return render_template("dashboard.html",
                           username=session.get("username"),
                           total_users=total_users,
                           total_predictions=total_predictions,
                           high_risk=high_risk,
                           recent_predictions=recent)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/customer/dashboard")
@customer_required
def customer_dashboard():
    db    = get_db()
    uid   = session["user_id"]
    total = db.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (uid,)).fetchone()[0]
    high  = db.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND risk_level='High'", (uid,)).fetchone()[0]
    recent = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 5", (uid,)
    ).fetchall()
    return render_template("customer_dashboard.html",
                           full_name=session.get("full_name"),
                           username=session.get("username"),
                           total_predictions=total,
                           high_risk=high,
                           recent_predictions=recent)


@app.route("/customer/predictions")
@customer_required
def my_predictions():
    db  = get_db()
    uid = session["user_id"]
    predictions = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC", (uid,)
    ).fetchall()
    return render_template("my_predictions.html",
                           full_name=session.get("full_name"),
                           predictions=predictions)


@app.route("/customer/profile")
@customer_required
def profile():
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    return render_template("profile.html", user=user)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ROUTES  (accessible to all, history saved when logged in)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html")

    missing = [f for f in feature_names if not request.form.get(f, "").strip()]
    if missing:
        return render_template("prediction.html",
                               error=f"Missing values for: {', '.join(missing)}. Please fill in all 14 CBC fields.")

    try:
        values = {f: float(request.form[f]) for f in feature_names}
    except ValueError as e:
        return render_template("prediction.html",
                               error=f"Invalid numeric value: {e}. All CBC fields must be numbers.")

    X            = pd.DataFrame([values], columns=feature_names)
    X_scaled     = scaler.transform(X)
    pred_index   = int(model.predict(X_scaled)[0])
    predicted_disease = str(label_encoder.inverse_transform([pred_index])[0])

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X_scaled)[0]
        confidence = f"{float(np.max(proba)) * 100:.1f}%"
    else:
        confidence = "N/A"

    info = DIAGNOSIS_INFO.get(predicted_disease, DEFAULT_INFO)

    # Save to history if a customer is logged in
    if session.get("role") == "customer":
        patient_name = request.form.get("patientName", "").strip() or "—"
        db = get_db()
        db.execute(
            """INSERT INTO predictions (user_id, patient_name, predicted_disease,
               confidence, risk_level, created_at) VALUES (?,?,?,?,?,?)""",
            (session["user_id"], patient_name, predicted_disease,
             confidence, info["risk_level"],
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()

    return render_template(
        "result.html",
        predicted_disease = predicted_disease,
        confidence        = confidence,
        risk_level        = info["risk_level"],
        recommendation    = info["recommendation"],
        suggested_tests   = info["suggested_tests"],
        lifestyle         = info["lifestyle"],
        doctor_note       = info["doctor_note"],
    )


@app.route("/result")
def result():
    return redirect(url_for("prediction"))


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
