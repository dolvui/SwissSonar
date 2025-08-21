from pymongo import MongoClient
from datetime import datetime
import streamlit as st
import os

# Load secrets
try:
    MONGO_URI = st.secrets["mongo"]["uri"]
    DB_NAME = st.secrets["mongo"]["db_name"]
except Exception:
    MONGO_URI = os.environ.get("MONGO_URI")
    DB_NAME = os.environ.get("DB_NAME")

# Init client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
models = db.models   # new collection


def init_db():
    """Ensure indexes exist for models collection."""
    models.create_index("name", unique=True)


def insert_models(data: dict):
    """Insert full model entry with metrics + timestamp."""
    data["timestamp"] = data.get("timestamp", datetime.utcnow())
    models.insert_one(data)


def insert_model_github(data: dict):
    """
    Insert a new model when triggered from Streamlit training.
    Returns the inserted model's ID.
    """
    data["timestamp"] = datetime.utcnow()
    result = models.insert_one(data)
    return str(result.inserted_id)


def fetch_models():
    """Fetch all models."""
    return list(models.find({}, {"_id": 0}))


def fetch_models_by_name(name: str):
    """Fetch models filtered by name."""
    return list(models.find({"name": name}, {"_id": 0}))


def fetch_models_by_id(id: str):
    """Fetch model by its MongoDB _id (string)."""
    from bson import ObjectId
    doc = models.find_one({"_id": ObjectId(id)}, {"_id": 0})
    return [doc] if doc else []


def update_model_benchmark(id: str, mse: float, mae: float):
    """Update benchmark metrics of a model."""
    from bson import ObjectId
    models.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"mse": mse, "mle": mae}}
    )


def remove_model_by_path(path: str):
    """Remove a model entry by its path."""
    models.delete_one({"path": path})
