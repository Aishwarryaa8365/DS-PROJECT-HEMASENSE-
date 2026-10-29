import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TMPL = os.path.join(ROOT, "frontend", "templates")

# Public templates that incorrectly link to admin-only /dashboard
public = [
    "index.html", "login.html", "signup.html",
    "prediction.html", "result.html",
    "diseases.html", "about.html", "contact.html"
]

OLD = "url_for('dashboard')"
NEW = "url_for('customer_dashboard') if session.get('role') == 'customer' else url_for('dashboard')"

for name in public:
    path = os.path.join(TMPL, name)
    txt = open(path, encoding="utf-8").read()
    if OLD in txt:
        txt = txt.replace(OLD, NEW)
        open(path, "w", encoding="utf-8").write(txt)
        print(f"Fixed: {name}")
    else:
        print(f"No change: {name}")

print("Done.")
