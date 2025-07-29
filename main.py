import coingeckoAPI
import swissUpdate
from onlineTrend import get_google_trend_score, count_reddit_mentions, youtube_search_count
from graph import plot_top_mentions
from sqliteDB import init_db, insert_token
#from alert import send_alert
from anomaly_detector import detect_anomalies, check_and_alert

init_db()

tokens, new_ids = swissUpdate.get_swissUpadte()

enriched_tokens = coingeckoAPI.fetch_market_data_fast(tokens, new_ids)

for token in enriched_tokens:
    print(f"Processing {token.name}")

    token.trend_score = get_google_trend_score(token.name)

    try:
        token.reddit_mentions = count_reddit_mentions(token.name)
    except:
        print('reddit failed')
        token.reddit_mentions = 0

    token.youtube_mentions = youtube_search_count(token.name)

    insert_token(token.dict_data())

for token in enriched_tokens:
    anomalies = detect_anomalies(token.name)
    if anomalies:
        check_and_alert(token.name, anomalies)

#make me lag for no reason
#plot_top_mentions()
