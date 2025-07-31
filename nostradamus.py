import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

def train_rnn(data, window=10, epochs=2000, lr=0.01):
    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)

    mean, std = prices.mean(), prices.std()
    prices_norm = (prices - mean) / std

    X, Y = build_sequences(prices_norm, window)
    X_tensor = torch.tensor(X).unsqueeze(2)  # [batch, time, 1]
    Y_tensor = torch.tensor(Y).unsqueeze(1)  # [batch, 1]

    model = PriceRNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        # if epoch % 500 == 0:
        #     print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    # Predict next value
    model.eval()
    out = None
    out_actual = None
    with torch.no_grad():
        out = model(X_tensor).squeeze().numpy()
        out_actual = Y_tensor.squeeze().numpy()

    last_seq = torch.tensor(prices_norm[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    next_norm = model(last_seq).item()
    next_price = next_norm * std + mean

    # Evaluation
    predicted_norm = model(X_tensor).detach().squeeze().numpy()
    predicted = predicted_norm * std + mean
    actual = Y_tensor.squeeze().numpy() * std + mean

    mse = np.mean((predicted - actual) ** 2)
    # print(f"\n🔮 Next predicted price: {next_price:.2f}")
    # print(f"📉 MSE on training set: {mse:.4f}")

    # Plot
    # plt.figure(figsize=(10, 4))
    # plt.plot(actual, label="Actual")
    # plt.plot(predicted, label="Predicted")
    # plt.title("Training Fit")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return out[-100:], out_actual[-100:], next_price, model

import matplotlib.pyplot as plt

def generate_prediction_plot(data, window=10, future_steps=20):
    predicted_train, actual_train, next_price, model = train_rnn(data, window=window)

    prices = np.array([float(e[1]) for e in data['prices']], dtype=np.float32)
    mean, std = prices.mean(), prices.std()
    prices_norm = (prices - mean) / std

    # Start with last known sequence
    future_input = prices_norm[-window:].tolist()
    future_preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            seq = torch.tensor(future_input[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_norm = model(seq).item()
            future_preds.append(next_norm)
            future_input.append(next_norm)  # Append prediction to the sequence

    # Denormalize
    future_preds = np.array(future_preds) * std + mean

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Actual Prices", color="blue")
    #plt.plot(np.arange(window, window + len(predicted_train)), predicted_train, label="Predicted (Train)", color="orange")
    plt.plot(np.arange(len(prices), len(prices) + future_steps), future_preds, label=f"Next {future_steps} Predictions", color="red", linestyle="--")
    plt.title("Price Prediction RNN")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.show()
    plt.close()
    return buf,predicted_train, actual_train, next_price, model
