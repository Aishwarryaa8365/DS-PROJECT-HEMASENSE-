import sys
sys.path.insert(0, "backend")
from backend.app import app, init_db

init_db()
c = app.test_client()

def check(label, r, expected_status, expected_loc=None):
    ok = r.status_code == expected_status
    if expected_loc:
        ok = ok and (expected_loc in r.headers.get("Location", ""))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label} -> {r.status_code} {r.headers.get('Location','')}")

# Public pages
check("GET /",           c.get("/"),           200)
check("GET /login",      c.get("/login"),       200)
check("GET /signup",     c.get("/signup"),      200)
check("GET /diseases",   c.get("/diseases"),    200)
check("GET /about",      c.get("/about"),       200)
check("GET /contact",    c.get("/contact"),     200)
check("GET /prediction", c.get("/prediction"),  200)

# Admin login
r = c.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
check("POST /login admin", r, 302, "/dashboard")

# Admin dashboard (with session)
with c.session_transaction() as sess:
    sess["user_id"] = 1
    sess["username"] = "admin"
    sess["full_name"] = "Administrator"
    sess["role"] = "admin"
check("GET /dashboard (admin)", c.get("/dashboard"), 200)

# Customer login - fresh client
c2 = app.test_client()
r = c2.post("/login", data={"username": "testuser", "password": "test123"}, follow_redirects=False)
check("POST /login customer", r, 302, "/customer/dashboard")

# Wrong password - fresh client
c3 = app.test_client()
r = c3.post("/login", data={"username": "admin", "password": "wrongpass"}, follow_redirects=False)
check("POST /login wrong pw", r, 200)  # stays on login page with error

# Signup new user - fresh client
import time
c4 = app.test_client()
r = c4.post("/signup", data={
    "full_name": "New User", "email": f"new{int(time.time())}@test.com",
    "username": f"newu{int(time.time())}", "password": "pass123", "confirm_password": "pass123"
}, follow_redirects=False)
check("POST /signup new user", r, 302, "/login")

print("\nAll tests complete.")
