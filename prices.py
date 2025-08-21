import yfinance as yf
import requests

def get_price(provider, symbol):
    if provider == "crypto":
        try:
            # you said you already have a ticker -> coingecko_id mapping
            # Example: BTC -> bitcoin
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
            r = requests.get(url).json()
            return r.get(symbol.lower(), {}).get("usd", 0.0)
        except:
            return 1.0
    elif provider == "stock":
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            return float(data["Close"].iloc[-1])
        except:
            return 1.0
    elif provider == "forex":
        # Stub: implement using free FX API
        return 1.0
    return 0.0
