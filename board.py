# board.py
from pymongo import MongoClient
from datetime import datetime
import streamlit as st
import os

try:
    MONGO_URI = st.secrets["mongo"]["uri"]
    DB_NAME = st.secrets["mongo"]["db_name"]
except Exception:
    MONGO_URI = os.environ.get("MONGO_URI")
    DB_NAME = os.environ.get("DB_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
boards = db.boards

def get_board(user: str):
    """Return the user's board (or create one if not exists)."""
    board = boards.find_one({"user": user})
    if not board:
        board = {"user": user, "rubricks": [], "created_at": datetime.utcnow()}
        boards.insert_one(board)
    return board

def add_rubrick(user: str, rubrick: str):
    boards.update_one(
        {"user": user},
        {"$push": {"rubricks": {"name": rubrick, "items": []}}},
        upsert=True
    )

def delete_rubrick(user: str, rubrick: str):
    boards.update_one(
        {"user": user},
        {"$pull": {"rubricks": {"name": rubrick}}}
    )

def add_item(user: str, rubrick: str, item: str):
    boards.update_one(
        {"user": user, "rubricks.name": rubrick},
        {"$push": {"rubricks.$.items": item}}
    )

def delete_item(user: str, rubrick: str, item: str):
    boards.update_one(
        {"user": user, "rubricks.name": rubrick},
        {"$pull": {"rubricks.$.items": item}}
    )
