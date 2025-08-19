import sqlite3
import shutil
from pathlib import Path

DB_PATH = Path("/tmp/models.db")

# Copy db from repo if not already in /tmp
if not DB_PATH.exists():
    repo_db = Path("models.db")
    if repo_db.exists():
        shutil.copy(repo_db, DB_PATH)

# One global connection (thread-safe mode ON)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")


def init_db():
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        path TEXT,
        days INTEGER,
        windows INTEGER,
        steps INTEGER,
        epochs INTEGER,
        lr REAL,
        hidden INTEGER,
        mse REAL,
        mle REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    cur.close()


def insert_models(data: dict):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO models (name,path,days,windows,steps,epochs,lr,hidden,mse,mle,timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (data["name"], data["path"], data["days"], data["windows"], data["steps"],
          data["epochs"], data["lr"], data["hidden"], data["mse"], data["mle"], data["timestamp"]))
    conn.commit()
    cur.close()


def insert_model_github(data: dict):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO models (name, path, days, windows, steps, epochs, lr, hidden)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (data["name"], data["path"], data["days"], data["windows"], data["steps"],
          data["epochs"], data["lr"], data["hidden"]))
    conn.commit()

    cur.execute("SELECT id FROM models WHERE name = ?", (data["name"],))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def fetch_models():
    cur = conn.cursor()
    cur.execute("SELECT * FROM models")
    rows = cur.fetchall()
    cur.close()
    return [dict(row) for row in rows]


def fetch_models_by_name(name: str):
    cur = conn.cursor()
    cur.execute("SELECT * FROM models WHERE name = ?", (name,))
    rows = cur.fetchall()
    cur.close()
    return [dict(row) for row in rows]


def fetch_models_by_id(id: int):
    cur = conn.cursor()
    cur.execute("SELECT * FROM models WHERE id = ?", (id,))
    rows = cur.fetchall()
    cur.close()
    return [dict(row) for row in rows]


def update_model_benchmark(id: int, mse: float, mae: float):
    cur = conn.cursor()
    cur.execute("UPDATE models SET mse = ?, mle = ? WHERE id = ?", (mse, mae, id))
    conn.commit()
    cur.close()


def remove_model_by_path(path: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE path = ?", (path,))
    conn.commit()
    cur.close()
