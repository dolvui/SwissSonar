import sqlite3
import matplotlib.pyplot as plt


def plot_top_mentions():
    conn = sqlite3.connect("crypto_tracker.db")
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT name, trend_score , reddit_mentions , youtube_mentions
                   FROM tokens
                   WHERE timestamp >= datetime('now', '-1 day')
                   ORDER BY trend_score DESC
                       LIMIT 100
                   """)
    rows = cursor.fetchall()

    combined = [(row[0], (row[1] + row[2] + row[3]) / 3) for row in rows]

    combined.sort(key=lambda x: x[1], reverse=True)

    names = [item[0] for item in combined]
    mentions = [item[1] for item in combined]

    plt.figure(figsize=(100, 60))
    plt.bar(names, mentions, color='skyblue')
    plt.title("Top 100 trend_score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
