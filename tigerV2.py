import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

save = "C:\\Users\\nghidalia\\PycharmProjects\\SwissSonar\\models\\"

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_steps=5, dropout=0.2):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_steps)
        self.tanh = nn.Tanh()  # Keep predictions in [-1, 1]

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.tanh(out)

def build_sequences(data, window, steps_ahead=5):
    """ Build multi-step sequences for prediction """
    X, Y = [], []
    for i in range(len(data) - window - steps_ahead):
        X.append(data[i:i + window])
        Y.append(data[i + window:i + window + steps_ahead])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_lstm(data, window=10, steps_ahead=50, epochs=500, lr=0.01):
    # Convert prices to log returns
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)
    returns = np.diff(np.log(prices))  # log returns
    mean, std = returns.mean(), returns.std()
    returns_norm = (returns - mean) / std

    X, Y = build_sequences(returns_norm, window, steps_ahead)
    X_tensor = torch.tensor(X).unsqueeze(2)  # [batch, time, 1]
    Y_tensor = torch.tensor(Y)  # [batch, steps_ahead]

    model = PriceLSTM(input_size=1, hidden_size=32, output_steps=steps_ahead, dropout=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch_graph = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        epoch_graph.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.8f}")

    # Predict next steps from last window
    model.eval()
    last_seq = torch.tensor(returns_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        next_returns_norm = model(last_seq).squeeze().numpy()

    # Convert predicted returns back to prices
    next_returns = next_returns_norm * std + mean
    last_price = prices[-1]
    predicted_prices = [last_price]
    for r in next_returns:
        predicted_prices.append(predicted_prices[-1] * np.exp(r))
    predicted_prices = predicted_prices[1:]  # Remove initial

    return predicted_prices, model ,prices,epoch_graph

from coingeckoAPI import fetch_token_price

# data = fetch_token_price("osmosis-allbtc",days=180)
# predicted_prices, model , prices, epoch_graph = train_lstm(data, lr=0.001, window=15, epochs=500000)
#
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# torch.save(model.state_dict(), save + f"tigerV2_{timestamp}.pt")
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 5))
# plt.plot(epoch_graph, label="los value", color="black")
# plt.title(f"loss function")
# plt.xlabel("Time Steps")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.close()
#
# plt.figure(figsize=(12, 5))
# plt.plot(prices, label="Actual Prices", color="blue")
#
# plt.plot(np.arange(len(prices), len(prices) + len(predicted_prices)),predicted_prices , label=f"Next {len(predicted_prices)} Predictions", color="red", linestyle="--")
# plt.title(f"Price Prediction RNN for BTC")
# plt.xlabel("Time Steps")
# plt.ylabel("Price")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.close()

def run_model_and_plot(model_path, data, window=15, steps_ahead=50, days=180):
    """
    Load a trained model, run predictions on new data, and generate a plot.
    """
    # 1️⃣ Load trained model
    model = PriceLSTM(input_size=1, hidden_size=32, output_steps=steps_ahead)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2️⃣ Fetch new market data
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

    # Convert prices to log returns (same preprocessing as training)
    returns = np.diff(np.log(prices))
    mean, std = returns.mean(), returns.std()
    returns_norm = (returns - mean) / std

    # 3️⃣ Prepare last sequence
    last_seq = torch.tensor(returns_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    # Predict future steps
    with torch.no_grad():
        next_returns_norm = model(last_seq).squeeze().numpy()

    # Convert predictions back to prices
    next_returns = next_returns_norm * std + mean
    last_price = prices[-1]
    predicted_prices = [last_price]
    for r in next_returns:
        predicted_prices.append(predicted_prices[-1] * np.exp(r))
    predicted_prices = np.array(predicted_prices[1:])

    import matplotlib.pyplot as plt
    # 4️⃣ Generate plot
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