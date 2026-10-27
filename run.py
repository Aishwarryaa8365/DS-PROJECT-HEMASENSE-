"""
HemaSense – entry point.
Run from the project root:  python run.py
"""
import sys
import os

# Ensure backend/ is on the path so relative imports inside app.py work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.app import app, init_db

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
