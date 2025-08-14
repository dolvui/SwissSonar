import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

save = "C:\\Users\\nghidalia\\PycharmProjects\\SwissSonar\\models\\"

cryptos = {
    #"ADA": "cardano",
    "BNB": "binance-coin-wormhole",
    "BTC": "osmosis-allbtc",
    #"ETH": "the-ticker-is-eth",
    "SOL": "wrapped-solana"
}

class PriceRNN(nn.Module):
    def __init__(self):
        super(PriceRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def build_sequences(data, window):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        Y.append(data[i + window])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def build_multi_crypto_dataset(fetch_fn, days=90, window=10):
    X_all, Y_all = [], []
    for name, gecko_id in cryptos.items():
        data = fetch_fn(gecko_id, days=days)
        if data and data['prices']:
            print(f"OK {name}")
        prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

        mean, std = prices.mean(), prices.std()
        prices_norm = (prices - mean) / std

        X, Y = build_sequences(prices_norm, window)
        X_all.append(X)
        Y_all.append(Y)

    X_total = np.vstack(X_all)
    Y_total = np.concatenate(Y_all)
    return X_total, Y_total

def train_big_rnn(fetch_fn, days=90, window=10, epochs=50, lr=0.01):
    X_total, Y_total = build_multi_crypto_dataset(fetch_fn, days, window)

    X_tensor = torch.tensor(X_total).unsqueeze(2)  # [batch, time, 1]
    Y_tensor = torch.tensor(Y_total).unsqueeze(1)  # [batch, 1]

    model = PriceRNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch_graph = []
    loss_graph = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        epoch_graph.append(epoch)
        loss_graph.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.8f}")

    return model,(epoch_graph,loss_graph)

def predict_next_prices(model, fetch_fn, gecko_id, days=90, window=10, future_steps=15):
    data = fetch_fn(gecko_id, days=days)
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)
    mean, std = prices.mean(), prices.std()
    prices_norm = (prices - mean) / std

    future_input = prices_norm[-window:].tolist()
    future_preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            seq = torch.tensor(future_input[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_norm = model(seq).item()
            future_preds.append(next_norm)
            future_input.append(next_norm)

    future_preds = np.array(future_preds) * std + mean
    return future_preds,prices

from coingeckoAPI import fetch_token_price

name = "BTC"
big_model,(epoch_graph,loss_graph) = train_big_rnn(fetch_token_price, days=180, window=15, epochs=15000)

future_preds, prices = predict_next_prices(big_model, fetch_token_price, cryptos[f"{name}"])
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(big_model.state_dict(), save + f"tiger_{timestamp}.pt")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(loss_graph, label="loss function", color="black")
plt.title(f"loss during training")
plt.show()
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(prices, label="Actual Prices", color="blue")

plt.plot(np.arange(len(prices), len(prices) + len(future_preds)), future_preds, label=f"Next {len(future_preds)} Predictions", color="red", linestyle="--")
plt.title(f"Price Prediction RNN for {name}")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()