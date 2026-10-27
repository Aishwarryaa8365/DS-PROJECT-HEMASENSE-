import os

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY   = os.environ.get("SECRET_KEY", "hemasense-dev-secret-2025")
DB_PATH      = os.path.join(ROOT_DIR, "database", "hemasense.db")
MODEL_DIR    = os.path.join(ROOT_DIR, "ml", "saved_models")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_DIR   = os.path.join(ROOT_DIR, "frontend", "static")
