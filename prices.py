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
    symbols = [f'{s["symbol"]}' for s in stocks]
    #st.write(symbols)
    # Download last closing prices
    try:
        #df = yf.Ticker(symbols)
        #st.write(df)
        #.info['regularMarketPrice']
        df = yf.download(" ".join(symbols))['Close']
        #st.write(df)
    except Exception as e:
        st.error(f"Download error: {e}")
        df = {}

    st.write(df)
    for stock in stocks:
        if not stock:
            pass
        else:
            try:
                price = df[f'{stock["symbol"]}'].iloc[-1]
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