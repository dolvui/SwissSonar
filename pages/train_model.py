import streamlit as st
import pandas as pd
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from github_pusher import push_model_to_github, delete_model_from_github
from pathlib import Path
from coingeckoAPI import fetch_token_price

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

cryptos_available = {
    "BNB": "binance-coin-wormhole",
    "BTC": "osmosis-allbtc",
    "SOL": "wrapped-solana"
}

def benchmark_model(model, norms, crypto_id, days, window, steps_ahead):
    data = fetch_token_price(crypto_id, days=days)
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

    mean, std = norms.get(crypto_id, (prices.mean(), prices.std()))
    prices_norm = (prices - mean) / std

    # Last window for prediction
    last_window = torch.tensor(prices_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    model.eval()
    with torch.no_grad():
        pred_norm = model(last_window).numpy().flatten()

    # Unnormalize
    pred_prices = pred_norm * std + mean
    actual_prices = prices[-steps_ahead:]

    mse = np.mean((pred_prices - actual_prices) ** 2)
    mae = np.mean(np.abs(pred_prices - actual_prices))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(range(len(actual_prices)), actual_prices, label="Actual", color="black")
    ax.plot(range(len(pred_prices)), pred_prices, label="Predicted", color="red")
    ax.legend()
    ax.set_title(f"Benchmark for {crypto_id} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    return fig, mse, mae

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
                crypto_id = cryptos_available[selected_crypto_key]

                fig, mse, mae = benchmark_model(model_path, crypto_id, days, window, steps_ahead, norms)
                st.pyplot(fig)
                st.write(f"**MSE:** {mse:.4f} | **MAE:** {mae:.4f}")
        with col3:
            if st.button("🗑 Delete", key=f"del_{model_path}"):
                delete_model_from_github(model_path.name)
                st.warning(f"Deleted `{model_path.name}`")
                st.rerun()

# Crypto selection
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

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


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
    norms = {}

    for name, gecko_id in cryptos.items():
        data = []
        try:
            data = fetch_fn(gecko_id, days=days)
        except Exception as e:
            print(e)
        prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)
        mean, std = prices.mean(), prices.std()
        norms[name] = (mean, std)

        prices_norm = (prices - mean) / std
        X, Y = build_sequences(prices_norm, window, steps_ahead)
        X_all.append(X)
        Y_all.append(Y)

    X_total = np.vstack(X_all)
    Y_total = np.concatenate(Y_all)
    return X_total, Y_total, norms

def train_model():
    X, Y, norms = build_multi_crypto_dataset(fetch_token_price,
        {k: cryptos_available[k] for k in selected_cryptos}, days, window, steps_ahead)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    X_train_tensor = torch.tensor(X_train).unsqueeze(2)
    Y_train_tensor = torch.tensor(Y_train)
    X_test_tensor = torch.tensor(X_test).unsqueeze(2)
    Y_test_tensor = torch.tensor(Y_test)

    model = PriceLSTM(input_size=1, hidden_size=hidden_size, output_steps=steps_ahead)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, Y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % max(1, epochs // 100) == 0:
            progress_bar.progress((epoch + 1) / epochs)
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_tensor), Y_test_tensor).item()

    return model, losses, test_loss, norms

# =========================
# RUN TRAINING
# =========================
if st.button("🚀 Start Training"):
    model, losses, test_loss, norms = train_model()
    st.write(f"Test Loss: {test_loss:.4f}")
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
        norms_path = f"models/tigerV2_{model_name}_norms.npy"
        torch.save(model.state_dict(), model_path)
        np.save(norms_path, norms)
        st.success(f"Model saved as `tigerV2_{model_name}.pt`")
        push_model_to_github(model_path, commit_msg=f"Add model {model_name} - {datetime.datetime.now()}")
        push_model_to_github(norms_path, commit_msg=f"Add norms for {model_name} - {datetime.datetime.now()}")