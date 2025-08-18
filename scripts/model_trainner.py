import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from coingeckoAPI import fetch_token_price
from github_pusher import push_model_to_github

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

def benchmark_model(model_path, crypto_id, days, window, steps_ahead,norms):

    data = fetch_token_price(crypto_id, days=days)
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

    mean, std = norms.get(crypto_id, (prices.mean(), prices.std()))
    prices_norm = (prices - mean) / std

    # Last window for prediction
    last_window = torch.tensor(prices_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    model = PriceLSTM(input_size=1, hidden_size=32, output_steps=steps_ahead)
    model.load_state_dict(torch.load(model_path))
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

def build_sequences(data, window, steps_ahead):
    X, Y = [], []
    for i in range(len(data) - window - steps_ahead):
        X.append(data[i:i + window])
        Y.append(data[i + window:i + window + steps_ahead])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def build_multi_crypto_dataset(fetch_fn, cryptos, days, window, steps_ahead):
    X_all, Y_all = [], []
    norms = {}

    for name, gecko_id in cryptos.items():
        data = []
        try:
            data = fetch_fn(gecko_id, days=days)
            data['prices']
        except Exception as e:
            print(e)
            print(data)
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

selected_cryptos = {"BTC": "osmosis-allbtc"}

def train_model(name,days, window, steps_ahead, epochs, learning_rate, hidden_size):
    X, Y, norms = build_multi_crypto_dataset(fetch_token_price,selected_cryptos, days, window, steps_ahead)

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
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, Y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_tensor), Y_test_tensor).item()

    return model, losses, test_loss, norms

def run_model_and_plot(model_path, data, window=15, steps_ahead=50):
    """
    Load a trained model, run predictions on new data, and generate a plot.
    """
    #TODO call to databse to retrieve the right param
    model = PriceLSTM(input_size=1, hidden_size=32, output_steps=steps_ahead)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

    returns = np.diff(np.log(prices))
    mean, std = returns.mean(), returns.std()
    returns_norm = (returns - mean) / std

    last_seq = torch.tensor(returns_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    with torch.no_grad():
        next_returns_norm = model(last_seq).squeeze().numpy()

    next_returns = next_returns_norm * std + mean
    last_price = prices[-1]
    predicted_prices = [last_price]
    for r in next_returns:
        predicted_prices.append(predicted_prices[-1] * np.exp(r))
    predicted_prices = np.array(predicted_prices[1:])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Actual Prices", color="blue")
    plt.plot(np.arange(len(prices)-1, len(prices)-1 + len(predicted_prices)),
             predicted_prices, label="Predicted", color="red", linestyle="--")
    plt.title(f"Price Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf


def launch_train_model(id):
    from sqliteModels import fetch_models_by_id
    model_dict = fetch_models_by_id(id)
    model_dict = model_dict[0]
    name = model_dict["name"]
    model, losses, test_loss, norms = train_model(name,model_dict["days"], model_dict["windows"], model_dict["steps"], model_dict["epochs"], model_dict["lr"], model_dict["hidden"])
    model_path = f"models/tigerV2_{name}.pt"
    norms_path = f"models/tigerV2_{name}_norms.npy"
    torch.save(model.state_dict(), model_path)
    np.save(norms_path, norms)
    #files = [model_path, norms_path]
    #push_model_to_github(files, commit_msg=f"Add model {model_dict["name"]}-{id}-{datetime.datetime.now()}")