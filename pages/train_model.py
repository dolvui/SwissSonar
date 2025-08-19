from pathlib import Path

import numpy as np
import streamlit as st

from github_pusher import delete_model_from_github, request_training, push_db_to_github
from scripts.model_trainner import benchmark_model

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Train a Crypto Price Model", layout="wide", page_icon="💪")
st.title("SwissSonar Model Trainer")
st.write("Train custom LSTM models for multi-crypto price prediction.")

# =========================
# SIDEBAR - PARAMS
# =========================
st.sidebar.header("⚙️ Training Parameters")

# Dataset parameters
days = st.sidebar.number_input("Days of historical data", min_value=30, max_value=720, value=180)
window = st.sidebar.number_input("Window size", min_value=5, max_value=60, value=15)
steps_ahead = st.sidebar.number_input("Steps ahead", min_value=1, max_value=200, value=50)

# Model parameters
epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=100_000_000_000, value=500)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=0.001, format="%.5f")
hidden_size = st.sidebar.slider("Hidden size", min_value=8, max_value=256, value=32, step=8)

# Model save info
model_name = st.sidebar.text_input("Model Name", value="my_model")
save_model = st.sidebar.checkbox("Save model after training", value=True)

# Path to your models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Get list of model files
model_files = sorted([f for f in MODELS_DIR.iterdir() if f.is_file() and f.name.endswith(".pt")])

st.subheader("List of available models:")

from mongodb import fetch_token_24h

cryptos_available = {}
result = fetch_token_24h()
for e in result:
    cryptos_available[e['ticker']] = e['gecko_id']

if not model_files:
    st.info("No models found in the `models/` directory.")
else:
    for model_path in model_files:
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.markdown(f"**{model_path.name}**")

        with col2:
            if st.button("📊 Benchmark", key=f"bench_{model_path}"):
                norms_path = model_path.with_name(model_path.stem + "_norms.npy")
                if norms_path.exists():
                    norms = np.load(norms_path, allow_pickle=True).item()
                else:
                    norms = {}

                selected_crypto_key = st.selectbox("Select crypto for benchmark", options=cryptos_available.keys())

                crypto_id = "binance-coin-wormhole"

                fig, mse, mae = benchmark_model(model_path, crypto_id, days, window, steps_ahead, norms)
                st.pyplot(fig)
                st.write(f"**MSE:** {mse:.4f} | **MAE:** {mae:.4f}")
        with col3:
            if st.button("🗑 Delete", key=f"del_{model_path}"):
                delete_model_from_github(model_path.name)
                from sqliteModels import remove_model_by_path
                remove_model_by_path(model_path)
                st.warning(f"Deleted `{model_path.name}`")
                st.rerun()

# Crypto selection
selected_cryptos = st.sidebar.multiselect(
    "Select Cryptos",
    options=list(cryptos_available.keys()),
    default=list({"BTC" : "osmosis-allbtc"}.keys()),
)

# =========================
# PLACEHOLDERS FOR PLOTS
# =========================
col1, col2 = st.columns(2)
loss_chart_placeholder = col1.empty()
pred_chart_placeholder = col2.empty()

# =========================
# RUN TRAINING
# =========================
if st.button("🚀 Start Training"):
    st.write(f"model is gone for training come back later \n")
    st.write(f"Params are days : {days},window :{window},steps_ahead :{steps_ahead},epochs :{epochs},learning_rate :{learning_rate},hidden_size :{hidden_size},model_name :{model_name}")
    from sqliteModels import insert_model_github, init_db
    data = {
        "name" : model_name,
        "path" : f"models/tigerV2_{model_name}.pt",
        "days" : days,
        "windows" : window,
        "steps" : steps_ahead,
        "hidden": hidden_size,
        "epochs" : epochs,
        "lr" : learning_rate
        }
    id = -1
    try:
        id = insert_model_github(data)
    except:
        init_db()
        id = insert_model_github(data)
    push_db_to_github("/tmp/models.db")
    if id and id != -1:
        request_training(id,selected_cryptos)
    else:
        st.error("An error occurs !")