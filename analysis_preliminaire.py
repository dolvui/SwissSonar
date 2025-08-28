from scipy.stats import linregress
import numpy as np

def pct_change(series):
    if not series or series[0] == 0:
        return 0
    return ((series[-1] - series[0]) / series[0]) * 100

def volatility(series):
    return float(np.std(series)) if series else 0.0

def correlation(prices, volumes):
    if len(prices) < 2 or len(volumes) < 2:
        return 0
    return float(np.corrcoef(prices, volumes)[0, 1])

def price_slope(prices):
    if len(prices) < 2:
        return 0, 0
    x = np.arange(len(prices))
    slope, _, r_value, _, _ = linregress(x, prices)
    return float(slope), float(r_value**2)

def detect_peak(prices, tolerance=0.02):
    """
    Detect if current price is within `tolerance` of recent peak.
    Example: tolerance=0.02 means 2% below max.
    """
    if len(prices) < 10:
        return False

    max_price = max(prices[-50:])  # lookback 50 samples
    current_price = prices[-1]

    return current_price >= max_price * (1 - tolerance)


def analyse_token(name, cg_data, ticker):
    prices = [p[1] for p in cg_data.get('prices', [])]
    volumes = [v[1] for v in cg_data.get('total_volumes', [])]
    caps = [c[1] for c in cg_data.get('market_caps', [])]

    delta_price = pct_change(prices)
    vol_volatility = volatility(volumes)
    corr_pv = correlation(prices, volumes)
    slope, r2 = price_slope(prices)

    # Eviter division par 0
    avg_vol = np.mean(volumes) if volumes else 1
    rel_volatility = vol_volatility / avg_vol * 100

    # Calcul du score avec pondération
    score = (
        0.3 * abs(delta_price) +
        0.5 * rel_volatility +
        0.2 * abs(slope) * 100
    )

    # Baisse du score si très faible pattern
    if r2 < 0.15 and abs(corr_pv) < 0.2:
        score *= 0.5

    # Classification du signal
    if score > 25:
        signal = "🚨 Exceptional"
    elif score > 15:
        signal = "🔺 Strong"
    elif score > 8:
        signal = "🔹 Medium"
    else:
        signal = "⚪ Weak"

    # Commentaire qualitatif
    if slope > 0 and r2 > 0.5:
        comment = "📈 Clear uptrend"
    elif slope < 0 and r2 > 0.5:
        comment = "📉 Clear downtrend"
    elif rel_volatility > 30 and abs(corr_pv) > 0.6:
        comment = "🚨 Volume spike + correlation"
    else:
        comment = "🔍 No clear pattern"

    peak = detect_peak(prices)

    obj = {
        "name": name,
        "ticker": ticker,
        "delta_price_pct": round(delta_price, 2),
        "volatility_volume": round(vol_volatility, 2),
        "corr_price_volume": round(corr_pv, 2),
        "slope": round(slope, 2),
        "r2": round(r2, 2),
        "signal_score": round(score, 2),
        "signal": signal,
        "comment": comment,
        "peak": peak
    }

    report = f"""
    name : {name}  
    ticker: {ticker}  
    delta_price_pct: {round(delta_price, 2)}%  
    volatility_volume: {round(vol_volatility, 2)}  
    corr_price_volume: {round(corr_pv, 2)}  
    slope: {round(slope, 2)}  
    r2: {round(r2, 2)}  
    signal_score: {round(score, 2)}  
    signal: {signal}  
    comment: {comment}
    """

    return obj, report
