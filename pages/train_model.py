import streamlit as st
import pandas as pd
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from github_pusher import push_model_to_github
from pathlib import Path

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
epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=1_000_000, value=500)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=0.001, format="%.5f")
hidden_size = st.sidebar.slider("Hidden size", min_value=8, max_value=256, value=32, step=8)

# Model save info
model_name = st.sidebar.text_input("Model Name", value="my_model")
save_model = st.sidebar.checkbox("Save model after training", value=True)

# Path to your models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Get list of model files
model_files = sorted([f for f in MODELS_DIR.iterdir() if f.is_file()])

if not model_files:
    st.info("No models found in the `models/` directory.")
else:
    for model_path in model_files:
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            st.markdown(f"**{model_path.name}**")

        with col2:
            if st.button("📊 Benchmark", key=f"bench_{model_path}"):
                st.success(f"Benchmarking `{model_path.name}`...")
                # TODO: Call your benchmarking function here

        with col3:
            if st.button("🗑 Delete", key=f"del_{model_path}"):
                os.remove(model_path)
                st.warning(f"Deleted `{model_path.name}`")
                st.rerun()

# Crypto selection
cryptos_available = {
    "BNB": "binance-coin-wormhole",
    "BTC": "osmosis-allbtc",
    "SOL": "wrapped-solana"
}
selected_cryptos = st.sidebar.multiselect(
    "Select Cryptos",
    options=list(cryptos_available.keys()),
    default=list(cryptos_available.keys())
)

# =========================
# PLACEHOLDERS FOR PLOTS
# =========================
col1, col2 = st.columns(2)
loss_chart_placeholder = col1.empty()
pred_chart_placeholder = col2.empty()

# =========================
# MODEL TRAINING
# =========================

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_steps=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_steps)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.tanh(out)

def build_sequences(data, window, steps_ahead):
    X, Y = [], []
    for i in range(len(data) - window - steps_ahead):
        X.append(data[i:i + window])
        Y.append(data[i + window:i + window + steps_ahead])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def fetch_dummy_crypto_data(crypto_id, days):
    """Replace with your real API fetch."""
    t = np.linspace(0, days, days)
    prices = 100 + np.sin(t / 5) * 10 + np.random.normal(0, 2, size=len(t))
    return {"prices": [[i, p] for i, p in enumerate(prices)]}

def build_multi_crypto_dataset(fetch_fn, cryptos, days, window, steps_ahead):
    X_all, Y_all = [], []
    for name, gecko_id in cryptos.items():
        data = fetch_fn(gecko_id, days=days)
        prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)
        mean, std = prices.mean(), prices.std()
        prices_norm = (prices - mean) / std
        X, Y = build_sequences(prices_norm, window, steps_ahead)
        X_all.append(X)
        Y_all.append(Y)
    X_total = np.vstack(X_all)
    Y_total = np.concatenate(Y_all)
    return X_total, Y_total

def train_model():
    X, Y = build_multi_crypto_dataset(fetch_dummy_crypto_data, {k: cryptos_available[k] for k in selected_cryptos}, days, window, steps_ahead)
    X_tensor = torch.tensor(X).unsqueeze(2)
    Y_tensor = torch.tensor(Y)

    model = PriceLSTM(input_size=1, hidden_size=hidden_size, output_steps=steps_ahead)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % max(1, epochs // 100) == 0:
            progress_bar.progress((epoch + 1) / epochs)
    return model, losses

# =========================
# RUN TRAINING
# =========================
if st.button("🚀 Start Training"):
    model, losses = train_model()

    # Loss Plot
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(losses, label="Loss", color="black")
    ax_loss.set_title("Loss Curve")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.grid(True)
    ax_loss.legend()
    loss_chart_placeholder.pyplot(fig_loss)

    # Fake prediction plot
    fig_pred, ax_pred = plt.subplots()
    ax_pred.plot(np.sin(np.linspace(0, 10, 50)), label="Predicted", color="red")
    ax_pred.set_title("Example Predictions")
    ax_pred.grid(True)
    ax_pred.legend()
    pred_chart_placeholder.pyplot(fig_pred)

    if save_model:
        model_path = f"models/tigerV2_{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        st.success(f"Model saved as `tigerV2_{model_name}.pt`")
        push_model_to_github(model_path, commit_msg=f"Add model {model_name} - {datetime.datetime.now()}")