from pymongo import MongoClient
from datetime import datetime
import os, streamlit as st

try:
    MONGO_URI = st.secrets["mongo"]["uri"]
    DB_NAME = st.secrets["mongo"]["db_name"]
except:
    MONGO_URI = os.environ["MONGO_URI"]
    DB_NAME = os.environ["DB_NAME"]

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
boards = db.boards


def get_board(board_name: str):
    board = boards.find_one({"board_name": board_name})
    if not board:
        board = {"board_name": board_name, "rubricks": [], "created_at": datetime.now()}
        boards.insert_one(board)
    return board


def add_rubrick(board_name: str, rubrick: dict):
    boards.update_one(
        {"board_name": board_name},
        {"$push": {"rubricks": {"name": rubrick['name'], "provider" : rubrick['provider'],"items": []}}},
        upsert=True
    )


def delete_rubrick(board_name: str, rubrick: str):
    boards.update_one(
        {"board_name": board_name},
        {"$pull": {"rubricks": {"name": rubrick}}}
    )


def add_item(board_name: str, rubrick: str, symbol: str, buy_price: float, quantity: float):
    item = {"symbol": symbol, "buy_price": buy_price, "quantity": quantity}
    boards.update_one(
        {"board_name": board_name, "rubricks.name": rubrick},
        {"$push": {"rubricks.$.items": item}}
    )


def delete_item(board_name: str, rubrick: str, symbol: str):
    boards.update_one(
        {"board_name": board_name, "rubricks.name": rubrick},
        {"$pull": {"rubricks.$.items": {"symbol": symbol}}}
    )
