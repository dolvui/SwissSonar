import sqlite3

conn = sqlite3.connect("crypto_tracker.db")
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tokens (
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
    INSERT INTO tokens (name, ticker, category, is_new, trend_score, reddit_mentions, youtube_mentions,
        current_price, market_cap, volume_24h, change_24h)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['name'], data['ticker'], data['category'], data['is_new'], data['trend_score'],
        data['reddit_mentions'], data['youtube_mentions'], data['current_price'],
        data['market_cap'], data['volume_24h'], data['change_24h']
    ))
    conn.commit()
