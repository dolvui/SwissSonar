import yfinance as yf
import requests
from forex_python.converter import CurrencyRates
import streamlit as st

c = CurrencyRates()

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.split("-")[0]
    symbol = symbol.replace(" ", "")
    if symbol: #in ["MC", "OR", "AI", "BN", "DG"]:
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
    symbol = normalize_symbol(symbol)
    ticker = yf.Ticker(symbol)
    price = ticker.info['regularMarketPrice']
    return price

def get_price_forex(symbol,buy_price):
    try:
        return c.get_rate(symbol,'USD')
    except Exception as e:
        st.error('forex seems down !')
        return buy_price