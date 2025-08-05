import requests
from CryptoToken import Token
from datetime import datetime,timedelta

def fetch_market_data_fast(tokens, new_ids):
    # Step 1: Map tickers -> CoinGecko ID
    listing_url = "https://api.coingecko.com/api/v3/coins/list"
    all_coins = requests.get(listing_url).json()
    symbol_map = {coin['symbol'].lower(): coin['id'] for coin in all_coins}

    id_to_token = {}
    gecko_ids = []

    for token in tokens:
        sym = token.ticker.lower()
        gecko_id = symbol_map.get(sym)
        if not gecko_id:
            print(f"[WARN] Missing {token.ticker}")
            continue

        token.id = gecko_id
        gecko_ids.append(gecko_id)
        id_to_token[gecko_id] = token
        token.is_new = token.name in new_ids

    # Step 2: Grouped API call
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ','.join(gecko_ids),
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }

    response = requests.get(url, params=params).json()
    rep : [ Token ] = []
    # Step 3: Assign value
    for gid, data in response.items():
        token = id_to_token[gid]
        # here store the token id from gecko for futur use as token.id = .....
        token.current_price = data.get("usd")
        token.market_cap = data.get("usd_market_cap")
        token.volume_24h = data.get("usd_24h_vol")
        token.change_24h = data.get("usd_24h_change")
        rep.append(token)

    return rep

def fetch_token_price(name,days=7):
    start = datetime.now() - timedelta(days=days)
    end = datetime.now()

    url = f"https://api.coingecko.com/api/v3/coins/{name}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": f"{int(start.timestamp())}",
        "to": f"{int(end.timestamp())}",
        "precision": "5"
    }
    response = requests.get(url, params=params).json()
    return response