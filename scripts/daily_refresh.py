import coingeckoAPI
import swissUpdate
from mongodb import upsert_tokens_entry
from onlineTrend import fetch_online_trend

def refresh_coins():
    tokens, new_ids = swissUpdate.get_swissUpadte()
    enriched_tokens = coingeckoAPI.fetch_market_data_fast(tokens, new_ids)
    full_tokens = fetch_online_trend(enriched_tokens)
    upsert_tokens_entry(full_tokens)
