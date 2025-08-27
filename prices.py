import math
import numpy as np
from scipy.stats import linregress
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
    "United States" : "",
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
    "Russia Federation": ".ME",      # Moscow Exchange
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

@st.cache_data(ttl=3600)
def get_price_stocks(stocks):
    ret = []
    symbols = [yahoo_symbol(s['symbol'], s['country']) for s in stocks]
    try:
        df = yf.download(" ".join(symbols))['Close']
    except Exception as e:
        st.error(f"Download error: {e}")
        df = {}
    for stock in stocks:
        if not stock:
            pass
        else:
            try:
                price = df[yahoo_symbol(stock['symbol'],stock['country'])].iloc[-1]
            except:
                price = math.nan

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

def analyse_stock(row, period="6mo", interval="1d"):
    symbol, country = row
    symbol = yahoo_symbol(symbol, country)
    st.info(f"Starting analysis of {symbol}")
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            return None, f"No data for {symbol}"

        # Ensure no NaN rows
        df = df.dropna()
        prices = df['Close']
        volumes = df['Volume']

        # --- Indicators ---
        # Moving averages
        ma20 = prices.rolling(20).mean()
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()

        # Bollinger Bands (20-day, 2 std)
        std20 = prices.rolling(20).std()
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)

        # Relative Strength Index (RSI)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # Linear regression slope (trend strength)
        slope_pct, r2 = 0.0, 0.0
        try:
            x = np.arange(len(prices))
            slope, intercept, r_value, _, _ = linregress(x, prices)
            slope_pct = slope / prices.iloc[0] * 100
            r2 = r_value**2
        except:
            pass

        # Volatility (% annualized)
        daily_ret = prices.pct_change().dropna()
        volatility = np.std(daily_ret) * np.sqrt(252) * 100 if not daily_ret.empty else 0.0

        # --- Signal Scoring ---
        score = 0

        st.write(ma20)
        st.write(ma50)
        st.write(ma200)
        # ma20_last = ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else None
        # ma50_last = ma50.iloc[-1] if not np.isnan(ma50.iloc[-1]) else None
        # ma200_last = ma200.iloc[-1] if not np.isnan(ma200.iloc[-1]) else None
        #
        # if ma20_last and ma50_last and ma200_last:
        #     if ma20_last > ma50_last > ma200_last:
        #         score += 20  # bullish trend
        #     elif ma20_last < ma50_last < ma200_last:
        #         score -= 20  # bearish trend

        if not np.isnan(upper_band.iloc[-1]) and not np.isnan(lower_band.iloc[-1]):
            if prices.iloc[-1] > upper_band.iloc[-1]:
                score += 15
            elif prices.iloc[-1] < lower_band.iloc[-1]:
                score -= 15

        rsi_last = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        if rsi_last > 70:
            score -= 10  # overbought
        elif rsi_last < 30:
            score += 10  # oversold

        if volatility > 40:
            score -= 5  # high risk

        if slope_pct > 0.05 and r2 > 0.5:
            score += 10
        elif slope_pct < -0.05 and r2 > 0.5:
            score -= 10

        # --- Classification ---
        if score >= 25:
            signal = "🚀 Strong Bullish"
        elif score >= 10:
            signal = "📈 Mild Bullish"
        elif score <= -25:
            signal = "⚠️ Strong Bearish"
        elif score <= -10:
            signal = "📉 Mild Bearish"
        else:
            signal = "🔍 Neutral / Sideways"

        # --- Comments ---
        comment = []
        if score > 0:
            comment.append("Uptrend bias")
        elif score < 0:
            comment.append("Downtrend bias")
        else:
            comment.append("No clear trend")

        if rsi_last > 70:
            comment.append("Overbought")
        elif rsi_last < 30:
            comment.append("Oversold")

        if volatility > 40:
            comment.append("High volatility")

        obj = {
            "symbol": symbol,
            "latest_price": round(prices.iloc[-1], 2),
            "slope_pct": round(slope_pct, 3),
            "r2": round(r2, 2),
            "volatility_pct": round(volatility, 2),
            "rsi": round(rsi_last, 2),
            "signal_score": score,
            "signal": signal,
            "comment": ", ".join(comment)
        }

        report = f"""
        Symbol: {symbol}  
        Latest price: {round(prices.iloc[-1], 2)}  
        Trend slope: {round(slope_pct, 3)}% (R²={round(r2, 2)})  
        Volatility: {round(volatility, 2)}% annualized  
        RSI(14): {round(rsi_last, 2)}  
        Signal Score: {score}  
        Signal: {signal}  
        Comment: {", ".join(comment)}
        """

        return obj, report

    except Exception as e:
        import sys
        exc_type, exc_obj, exc_traceback = sys.exc_info()
        line_number = exc_traceback.tb_lineno
        st.error(f"Exception occurred on line {line_number}")
        return None, f"Error analysing {symbol}: {e}"
