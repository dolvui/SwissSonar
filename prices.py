import yfinance as yf
import requests
from forex_python.converter import CurrencyRates

c = CurrencyRates()

def get_price_cryptocurrency(symbol):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        r = requests.get(url).json()
        return r.get(symbol.lower(), {}).get("usd", 0.0)
    except:
        return -1.0

def get_price_stock(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d")
    print(data['Open'])
    print(data['High'])
    print(data['Low'])
    print(data['Adj Close'])
    print(data['Close'])
    print(data['Volume'])
    return float(data["Close"].iloc[-1])

def get_price_forex(symbol):
    return c.get_rate(symbol, "USD")