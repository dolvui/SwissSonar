import yfinance as yf
import requests
from forex_python.converter import CurrencyRates

# Yahoo Finance stock symbol suffixes by country/exchange
YAHOO_SUFFIXES = {
    # Europe
    "France": ".PA",      # Paris
    "Germany": ".DE",     # Deutsche Börse
    "Sweden": ".ST",      # Stockholm
    "Finland": ".HE",     # Helsinki
    "Denmark": ".CO",     # Copenhagen
    "Norway": ".OL",      # Oslo
    "Italy": ".MI",       # Milan
    "Spain": ".MC",       # Madrid
    "Switzerland": ".SW", # Swiss Exchange
    "UK": ".L",           # London Stock Exchange
    "Netherlands": ".AS", # Amsterdam
    "Belgium": ".BR",     # Brussels
    "Portugal": ".LS",    # Lisbon
    "Austria": ".VI",     # Vienna
    "Poland": ".WA",      # Warsaw
    "Greece": ".AT",      # Athens
    "Turkey": ".IS",      # Istanbul

    # Americas
    "USA": "",            # no suffix
    "Canada": ".TO",      # Toronto
    "Brazil": ".SA",      # São Paulo
    "Mexico": ".MX",      # Mexico
    "Argentina": ".BA",   # Buenos Aires
    "Chile": ".SN",       # Santiago

    # Asia-Pacific
    "Japan": ".T",        # Tokyo
    "China": ".SS",       # Shanghai
    "Hong Kong": ".HK",   # Hong Kong
    "South Korea": ".KS", # KOSPI
    "Taiwan": ".TW",      # Taiwan
    "India": ".BO",       # Bombay
    "Singapore": ".SI",   # Singapore
    "Australia": ".AX",   # ASX
    "New Zealand": ".NZ", # NZX

    # Others
    "Russia": ".ME",      # Moscow Exchange
    "South Africa": ".JO" # Johannesburg
}

import streamlit as st

c = CurrencyRates()

def yahoo_symbol(symbol: str, country: str) -> str:
    """
    Convert raw stock symbol + country to Yahoo Finance symbol.
    """
    suffix = YAHOO_SUFFIXES.get(country, "")
    return f"{symbol}{suffix}"

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.split("-")[0]
    symbol = symbol.replace(" ", "")
    if symbol: #in ["MC", "OR", "AI", "BN", "DG"]:
        return f"{symbol}.PA"
    return symbol

def get_price_cryptocurrency(symbol,default=0.0):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        r = requests.get(url).json()
        return r.get(symbol.lower(), {}).get("usd", default)
    except:
        return default

def get_price_stock(symbol,default = 0.0):
    try:
        symbol = normalize_symbol(symbol)
        ticker = yf.Ticker(symbol)
        price = ticker.info['regularMarketPrice']
        return price
    except:
        return default

# def get_price_stocks(stocks):
#     symbols_sting = []
#     ret = []
#
#     for stock in stocks:
#         #print(stock)
#         #st.write(stock)
#         symbols_sting.append(normalize_symbol(f"{stock['symbol']}-{stock['name']}"))
#
#     prices = [0.0 for symbol in symbols_sting]
#     try:
#         prices = yf.Ticker(symbols_sting).info['regularMarketPrice']
#     except:
#         pass
#     for s,p in stocks,prices:
#         ret.append({ "name": s["name"], "symbol": s["symbol"] , "country": s["country"], "price" : p, "industries": s["industries"] })
#
#     return ret

def get_price_stocks(stocks):
    ret = []
    symbols = [yahoo_symbol(s['symbol'], s['country']) for s in stocks]
    try:
        df = yf.download(" ".join(symbols))['Close']
    except Exception as e:
        st.error(f"Download error: {e}")
        df = {}

    st.write(df)
    for stock in stocks:
        if not stock:
            pass
        else:
            try:
                price = df[yahoo_symbol(stock['symbol'],stock['country'])].iloc[-1]
            except:
                price = 0.0

            ret.append({
                "name": stock["name"],
                "symbol": stock["symbol"],
                "country": stock["country"],
                "price": price,
                "industries": stock["industries"],
            })
    return ret

def get_price_forex(symbol,buy_price):
    try:
        return c.get_rate(symbol,'USD')
    except Exception as e:
        st.error('forex seems down !')
        return buy_price