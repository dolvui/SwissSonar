import sqlite3
from datetime import datetime , timedelta

conn = sqlite3.connect("old_files/crypto_tracker.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tokens (
        gecko_id TEXT,
        name TEXT,
        ticker TEXT,
        category TEXT,
        is_new BOOLEAN,
        trend_score INTEGER,
        reddit_mentions INTEGER,
        youtube_mentions INTEGER,
        current_price REAL,
        market_cap REAL,
        volume_24h REAL,
        change_24h REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

def insert_token(data: dict):
    cursor.execute("""
    INSERT INTO tokens (gecko_id, name, ticker, category, is_new, trend_score, reddit_mentions, youtube_mentions,
        current_price, market_cap, volume_24h, change_24h)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['id'], data['name'], data['ticker'], data['category'], data['is_new'], data['trend_score'],
        data['reddit_mentions'], data['youtube_mentions'], data['current_price'],
        data['market_cap'], data['volume_24h'], data['change_24h']
    ))
    conn.commit()

def fetch_last_7_days(token_name : str):
    seven_days_ago = datetime.now() - timedelta(days=7)
    cursor.execute("""
                   SELECT *
                   FROM tokens
                   WHERE name = ? AND timestamp >= ?
                   ORDER BY timestamp ASC
                   """, (token_name, seven_days_ago.isoformat()))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def fetch_token_gecko():
    cursor.execute(""" SELECT DISTINCT gecko_id,ticker FROM tokens""")
    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def fetch_token_24h():
    t24h_ago = datetime.now() - timedelta(hours=24)
    cursor.execute("""
                   SELECT gecko_id,
                          name,
                          ticker,
                          category,
                          is_new,
                          trend_score,
                          reddit_mentions,
                          youtube_mentions,
                          current_price,
                          market_cap,
                          volume_24h,
                          change_24h
                   FROM tokens
                   WHERE timestamp >= ?
                     AND timestamp IN (
                       SELECT MAX (timestamp)
                       FROM tokens
                       WHERE timestamp >= ?
                       GROUP BY gecko_id
                       )
                   """, (t24h_ago.isoformat(), t24h_ago.isoformat()))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]