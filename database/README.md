# Database

The SQLite database `hemasense.db` is stored here during local development.

It is excluded from version control via `.gitignore` to protect user data.

## Initialize locally

The database is created automatically on first run:

```bash
python run.py
```

This calls `init_db()` which creates the `users` and `predictions` tables
and seeds the default admin account:

| Field    | Value      |
|----------|------------|
| Username | admin      |
| Password | admin123   |

Change the admin password after first login.
