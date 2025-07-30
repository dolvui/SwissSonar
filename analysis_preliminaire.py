from scipy.stats import linregress
import numpy as np

def pct_change(series):
    if not series:
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

def analyse_token(name, cg_data, ticker):
    """
    cg_data is the JSON/dict from CoinGecko containing:
      'prices', 'total_volumes', 'market_caps',
    each as lists of [timestamp, value]
    """
    prices = [p[1] for p in cg_data.get('prices', [])]
    volumes = [v[1] for v in cg_data.get('total_volumes', [])]
    caps = [c[1] for c in cg_data.get('market_caps', [])]

    delta_price = pct_change(prices)
    vol_volatility = volatility(volumes)
    corr_pv = correlation(prices, volumes)
    slope, r2 = price_slope(prices)

    # Signal scoring thresholds (à ajuster)
    score_components = [
        abs(delta_price),
        vol_volatility / (np.mean(volumes) if volumes else 1) * 100,
        abs(slope) * 100
    ]
    signal_score = sum(score_components) / len(score_components)

    if signal_score > 5:
        signal = "🔺 Strong"
    elif signal_score > 2:
        signal = "🔹 Medium"
    else:
        signal = "⚪ Weak"

    obj =  {
        "name": name,
        "ticker": ticker,
        "delta_price_pct": round(delta_price, 2),
        "volatility_volume": round(vol_volatility, 2),
        "corr_price_volume": round(corr_pv, 2),
        "slope": round(slope, 2),
        "r2": round(r2, 2),
        "signal_score": signal_score,
        "signal": signal
    }

    report = (f" name : {name} , \n \
        ticker: {ticker}, \n \
        delta_price_pct: {round(delta_price, 2) } ,\n \
        volatility_volume: {round(vol_volatility, 2) },\n \
        corr_price_volume: {round(corr_pv, 2) } ,\n \
        slope: {roun(slope, 2) },\n \
        r2: {round(r2, 2) },\n \
        signal_score: {signal_score},\n \
        signal: {signal} \n")

    return obj,report