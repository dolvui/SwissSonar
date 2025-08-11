from pymongo import MongoClient
from datetime import datetime, timedelta
from CryptoToken import Token
import streamlit as st

# client = MongoClient("localhost",27017)
# db = client.crypto_db
# tokens = db.tokens

MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db_name"]

client = MongoClient(MONGO_URI)
print(client[DB_NAME])
db = client[DB_NAME]
tokens = db.tokens
print(tokens)
def upsert_tokens_entry(data):
    for token in data:
        upsert_token_entry(token.dict_data())

def upsert_token_entry(data: dict):
    gecko_id = data['id']
    timestamp = datetime.now()

    market_entry = {
        "timestamp": timestamp,
        "current_price": data['current_price'],
        "market_cap": data['market_cap'],
        "volume_24h": data['volume_24h'],
        "change_24h": data['change_24h']
    }

    trend_entry = {
        "timestamp": timestamp,
        "reddit_mentions": data['reddit_mentions'],
        "trend_score": data['trend_score'],
        "youtube_mentions": data['youtube_mentions']
    }

    tokens.update_one(
        {"gecko_id": gecko_id},
        {
            "$setOnInsert": {
                "name": data['name'],
                "ticker": data['ticker'],
                "category": data['category'],
                "is_new": data['is_new']
            },
            "$push": {
                "market_data": market_entry,
                "online_signals": trend_entry
            }
        },
        upsert=True
    )

def entity_to_token(doc):
    t = Token(doc["name"], doc["ticker"], "", 0, doc["category"])
    t.id = doc["gecko_id"]
    t.current_price = doc["market_data"][-1]["current_price"]
    t.market_cap = doc["market_data"][-1]["market_cap"]
    t.volume_24h = doc["market_data"][-1]["volume_24h"]
    t.change_24h = doc["market_data"][-1]["change_24h"]
    t.is_new = doc.get("is_new", False)
    t.trend_score = doc["online_signals"][-1]["trend_score"]
    t.reddit_mentions = doc["online_signals"][-1]["reddit_mentions"]
    t.youtube_mentions = doc["online_signals"][-1]["youtube_mentions"]
    return t

def fetch_token_gecko():
    return list(tokens.find({}, {"gecko_id": 1, "ticker": 1, "_id": 0}))

def fetch_last_7_days(token_name):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    doc = tokens.find_one({"name": token_name})
    if not doc:
        return []

    # Combine market + online signals by timestamp
    history = []
    for m, o in zip(doc.get("market_data", []), doc.get("online_signals", [])):
        if m["timestamp"] >= seven_days_ago:
            entry = {
                "name": doc["name"],
                "gecko_id": doc["gecko_id"],
                "reddit_mentions": o.get("reddit_mentions", -1),
                "trend_score": o.get("trend_score", -1),
                "youtube_mentions": o.get("youtube_mentions", -1),
                "current_price": m.get("current_price"),
                "market_cap": m.get("market_cap"),
                "volume_24h": m.get("volume_24h"),
                "change_24h": m.get("change_24h"),
                "timestamp": m["timestamp"]
            }
            history.append(entry)

    return sorted(history, key=lambda x: x["timestamp"])

def fetch_token_24h():
    t24h_ago = datetime.now() - timedelta(hours=24)
    result = []

    for doc in tokens.find():
        if not doc.get("market_data"): continue

        latest_market = [m for m in doc["market_data"] if m["timestamp"] >= t24h_ago]
        latest_signal = [s for s in doc.get("online_signals", []) if s["timestamp"] >= t24h_ago]

        if latest_market and latest_signal:
            last_m = sorted(latest_market, key=lambda x: x["timestamp"])[-1]
            last_s = sorted(latest_signal, key=lambda x: x["timestamp"])[-1]

            result.append({
                "gecko_id": doc["gecko_id"],
                "name": doc["name"],
                "ticker": doc["ticker"],
                "category": doc["category"],
                "is_new": doc.get("is_new", False),
                "trend_score": last_s.get("trend_score", -1),
                "reddit_mentions": last_s.get("reddit_mentions", -1),
                "youtube_mentions": last_s.get("youtube_mentions", -1),
                "current_price": last_m.get("current_price"),
                "market_cap": last_m.get("market_cap"),
                "volume_24h": last_m.get("volume_24h"),
                "change_24h": last_m.get("change_24h"),
                "timestamp": last_m["timestamp"]
            })

    return result

def get_latest_online_trends(gecko_id, count=2):
    doc = tokens.find_one({"gecko_id": gecko_id}, {"online_signals": 1})
    if not doc or "online_signals" not in doc:
        return []

    sorted_signals = sorted(doc["online_signals"], key=lambda x: x["timestamp"], reverse=True)
    return sorted_signals[:count]