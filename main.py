import time
import coingeckoAPI
import nostradamus
import swissUpdate
from analysis_preliminaire import analyse_token
from anomaly_detector import detect_anomalies
from coingeckoAPI import fetch_token_price
from mongodb import upsert_tokens_entry, get_latest_online_trends
from onlineTrend import fetch_online_trend, compute_heuristics
from pdf_builder import *
from sqliteDB import init_db
from datetime import datetime
import streamlit

import streamlit as st
x = st.slider("Select a value")
st.write(x, "squared is", x * x)

def process_token(tokenScore):
    print("\n📊 Top 5 tokens du jour :")
    report_token = []
    i = 0

    for name, score, ticker in tokenScore[:5]:
        print(f"🔸 {ticker} → score : {score:.2f}")
        data = fetch_token_price(name, days=180)
        # buf,predicted, actual, next_pred, model = nostradamus.generate_prediction_plot(data)
        path = "C:\\Users\\nghidalia\\PycharmProjects\\SwissSonar\\models\\tigerV2_20250807_152739.pt"
        from tigerV2 import run_model_and_plot
        Bbuff = run_model_and_plot(path, data)
        _, report = analyse_token(name, data, ticker)

        report_token.append({
            "name": ticker,
            "report": report,
            "next_pred": None,
            "actual": None,
            "predicted": None,
            "buf": Bbuff,
        })

        if i != 1 and i % 3 == 0:
            time.sleep(60)
        i += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_multi_pdf(report_token, filename=f"top5_crypto_report_{timestamp}.pdf")

def sort_token(tokens):
    tokensScore = []
    heur = 1
    for token in tokens:
        score = detect_anomalies(token.name)
        data = get_latest_online_trends(token.id)

        if len(data) >= 2 :
            google_trend = data[0]['trend_score']
            youtube_mentions = data[0]['youtube_mentions']
            reddit_mentions = data[0]['reddit_mentions']
            previous_google = data[1]['trend_score']
            previous_youtube = data[1]['youtube_mentions']
            previous_reddit = data[1]['reddit_mentions']
            heur = compute_heuristics(google_trend, youtube_mentions, reddit_mentions, previous_google, previous_youtube, previous_reddit)

        if score :
            tokensScore.append((token.id, (score/300) * heur , token.ticker))
        else:
            tokensScore.append((token.id, heur, token.ticker))

    tokensScore.sort(key=lambda x: x[1], reverse=True)
    return tokensScore

def fetch_coins():
    init_db()

    swissUpdate.init()
    tokens, new_ids = swissUpdate.get_swissUpadte()

    enriched_tokens = coingeckoAPI.fetch_market_data_fast(tokens, new_ids)

    full_tokens = fetch_online_trend(enriched_tokens)

    upsert_tokens_entry(full_tokens)

    tokensScore = sort_token(full_tokens)

    process_token(tokensScore)

def from_database():

    from mongodb import fetch_token_24h
    from CryptoToken import entity_to_token

    result = fetch_token_24h()
    tokens = []
    for e in result:
        token = entity_to_token(e)
        tokens.append(token)

    tokensScore = sort_token(tokens)

    process_token(tokensScore)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='select which action need to make')
    parser.add_argument('--fetch-coins', metavar='boolean', required=False, help='fetch coins or only use in DB')
    parser.add_argument('--train', metavar='None', required=False, help='train a RNN')

    args = parser.parse_args()

    if args.fetch_coins:
        if args.fetch_coins == 'True':
            fetch_coins()
        else:
            from_database()
    if args.train:
        from tigerV2 import train
        train()