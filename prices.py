import yfinance as yf
import requests
from forex_python.converter import CurrencyRates
import streamlit as st
c = CurrencyRates()

yf.enable_debug_mode()

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.split("-")[0]
    symbol = symbol.replace(" ", "")
    if symbol in ["MC", "OR", "AI", "BN", "DG"]:
        return f"{symbol}"
    return symbol

def get_price_cryptocurrency(symbol):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        r = requests.get(url).json()
        return r.get(symbol.lower(), {}).get("usd", 0.0)
    except:
        return -1.0

def get_price_stock(symbol):
    symbol = normalize_symbol(symbol)
    st.info(symbol)
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d")
    # try:
    #     st.info(data['Open'].iloc[-1])
    # except:
    #     pass
    #
    # try:
    #     st.info(data['High'].iloc[-1])
    # except:
    #     pass
    #
    # try:
    #     st.info(data['Low'].iloc[-1])
    # except:
    #     pass
    # #st.info(data['Close'].iloc)
    #
    # try:
    #     st.info(data['Close'].iloc[-1])
    # except:
    #     pass
    #
    # try:
    #     st.info(data['Volume'].iloc[-1])
    # except:
    #     pass
    #print(data['Open'])
    #print(data['High'])
    #print(data['Low'])
    #print(data['Adj Close'])
    #print(data['Close'])
    #print(data['Volume'])
    return float(data["Close"].iloc[-1])

def get_price_forex(symbol):
    try:
        return c.get_rate(symbol, "USD")
    except:
        return 0.0