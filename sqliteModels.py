import sqlite3
import shutil
from pathlib import Path

DB_PATH = Path("/tmp/models.db")

if not DB_PATH.exists():
    repo_db = Path("models.db")
    if repo_db.exists():
        shutil.copy(repo_db, DB_PATH)

conn = sqlite3.connect(DB_PATH,check_same_thread=False)
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
cursor = conn.cursor()

def init_db():
    cursor.execute("""
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

def insert_models(data: dict):
    cursor.execute("""
    INSERT INTO models (name,path,days,windows,steps,epochs,lr,hidden,mse,mle,timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (data["name"], data["path"], data["days"], data["windows"], data["steps"],data["epochs"],data["lr"],data["hidden"],data["mse"],data["mle"],data["timestamp"]))
    conn.commit()

def insert_model_github(data: dict):
    cursor.execute("""INSERT INTO models (name, path, days, windows, steps, epochs, lr, hidden)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    """, (data["name"], data["path"], data["days"], data["windows"], data["steps"],data["epochs"],data["lr"],data["hidden"]))
    conn.commit()
    cursor.execute(""" SELECT id FROM models WHERE name = ?""", (data["name"],))
    rows = cursor.fetchone()
    return rows[0] if rows else None

def fetch_models():
    cursor.execute(""" SELECT * FROM models""")
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def fetch_models_by_name(name: str):
    cursor.execute(""" SELECT * FROM models WHERE name = ?""", (name,))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def fetch_models_by_id(id):
    cursor.execute(""" SELECT * FROM models WHERE id = ?""", (id,))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def update_model_benchmark(id,mse,mae):
    cursor.execute("""UPDATE models SET mse = ? , mle = ? WHERE id = ?""", (mse, mae, id))
    conn.commit()

def remove_model_by_path(path):
    cursor.execute("""DELETE FROM models WHERE path = ?""",(path,))
    conn.commit()