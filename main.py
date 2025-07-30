from datetime import timedelta

import coingeckoAPI
import swissUpdate
from onlineTrend import get_google_trend_score, count_reddit_mentions, youtube_search_count
from graph import plot_top_mentions
from sqliteDB import init_db, insert_token
#from alert import send_alert
from anomaly_detector import detect_anomalies, check_and_alert
from coingeckoAPI import fetch_token_price
from analysis_preliminaire import analyse_token


def main():
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

    tokensScore : [(str,int)] = []

    for token in enriched_tokens:
        score = detect_anomalies(token.name)
        if score:
            tokensScore.append( (token.id, score , token.ticker) )

    tokensScore.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 Top 5 tokens du jour :")
    for name, score , ticker in tokensScore[:5]:
        print(f"🔸 {ticker} → score : {score:.2f}")
        data = fetch_token_price(name)
        obj, report = analyse_token(name,data,ticker)
        print(report)

    #make me lag for no reason
    #plot_top_mentions()


main()
# name = "0x Protocol"
# data = fetch_token_price(name)
# report = analyse_token(name,data)
# print(report)

