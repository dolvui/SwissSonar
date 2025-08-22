import yfinance as yf
import requests
from forex_python.converter import CurrencyRates
import streamlit as st
c = CurrencyRates()

def normalize_symbol(symbol: str) -> str:
    if symbol in ["MC", "OR", "AI", "BN", "DG"]:
        return f"{symbol}.PA"
    return symbol

def get_price_cryptocurrency(symbol):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        r = requests.get(url).json()
        return r.get(symbol.lower(), {}).get("usd", 0.0)
    except:
        return -1.0

def get_price_stock(symbol):
    symbol = symbol.split("-")[0]
    symbol = symbol.replace(" ","")#normalize_symbol()
    st.info(symbol)
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d")
    print(data)
    #print(data['Open'])
    #print(data['High'])
    #print(data['Low'])
    #print(data['Adj Close'])
    #print(data['Close'])
    #print(data['Volume'])
    return 0.0#float(data["Adj Close"].iloc[-1])

def get_price_forex(symbol):
    try:
        return c.get_rate(symbol, "USD")
    except:
        return -1.0