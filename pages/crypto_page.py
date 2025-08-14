import pandas as pd
import streamlit as st
import coingeckoAPI
import swissUpdate
from CryptoToken import entity_to_token
from coingeckoAPI import fetch_token_price
from mongodb import fetch_token_24h
from mongodb import upsert_tokens_entry, get_latest_online_trends
from onlineTrend import fetch_online_trend, compute_heuristics
from analysis_preliminaire import analyse_token
from pathlib import Path

def tokens_heuristic(df_tokens):
    heuristics = []

    for token_id in df_tokens["id"]:
        data = get_latest_online_trends(token_id)

        if len(data) >= 2:
            google_trend = data[0]['trend_score']
            youtube_mentions = data[0]['youtube_mentions']
            reddit_mentions = data[0]['reddit_mentions']
            previous_google = data[1]['trend_score']
            previous_youtube = data[1]['youtube_mentions']
            previous_reddit = data[1]['reddit_mentions']

            heur = compute_heuristics(
                google_trend,
                youtube_mentions,
                reddit_mentions,
                previous_google,
                previous_youtube,
                previous_reddit
            )
        else:
            heur = 1  # default value

        heuristics.append(heur)

    # Add column to DataFrame
    df_tokens["heuristic"] = heuristics
    return df_tokens

result = fetch_token_24h()
tokens = []
times = []
for e in result:
    times.append(e["timestamp"])
    token = entity_to_token(e)
    tokens.append(token)

#tokensScore = sort_token(tokens)
df_tokens = pd.DataFrame([t.dict_data() for t in tokens])
# ========================
# FRONT PAGE LAYOUT
# ========================
st.set_page_config(page_title="Crypto Analysis Dashboard", layout="wide", page_icon="📊")

st.title("📊 Crypto Analysis Dashboard")

# ---- Dashboard Section ----
st.subheader("Dashboard Overview")

st.subheader("Actions")
actions1,actions2 = st.columns(2)

with actions1:
    if st.button("refresh") :
        tokens, new_ids = swissUpdate.get_swissUpadte()
        enriched_tokens = coingeckoAPI.fetch_market_data_fast(tokens, new_ids)
        full_tokens = fetch_online_trend(enriched_tokens)
        upsert_tokens_entry(full_tokens)
        df_tokens = pd.DataFrame([t.dict_data() for t in full_tokens])

with actions2:
    if st.button("Train my own model"):
        st.switch_page("pages/train_model.py")

df_tokens = tokens_heuristic(df_tokens)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tokens", len(df_tokens))
with col2:
    top_marketcap = df_tokens.loc[df_tokens["market_cap"].idxmax()]["name"]
    st.metric("Highest Market Cap", top_marketcap)
with col3:
    top_trend = df_tokens.loc[df_tokens["heuristic"].idxmax()]["name"]
    st.metric("Top Trend Token", top_trend)
with col4:
    st.metric("Last fetch", f"{max(times)}")

st.markdown("---")

# ---- Search & List Section ----
st.subheader("Token List & Analysis")

search_query = st.text_input("🔍 Search token by name or ticker").lower()

if search_query:
    filtered_df = df_tokens[
        df_tokens["name"].str.lower().str.contains(search_query)
        | df_tokens["ticker"].str.lower().str.contains(search_query)
        ]
else:
    filtered_df = df_tokens

# Display table
st.dataframe(
    filtered_df[["id","name", "ticker", "current_price", "volume_24h", "market_cap", "heuristic"]],
    use_container_width=True
)

# Create mapping from display name to ID
display_options = {
    f"{row['name']} ({row['ticker']})": row['id']
    for _, row in filtered_df.iterrows()
}

# Selectbox shows nice text, returns the corresponding ID
selected_display = st.selectbox("Select a token for analysis", list(display_options.keys()))

MODELS_DIR = Path("models")
model_files = sorted([f.name for f in MODELS_DIR.iterdir() if f.is_file() and f.name.endswith(".pt")])
selected_models = st.selectbox("Select a models",model_files)

selected_token = display_options[selected_display]

if st.button("🔎 Analyse"):
    st.success(f"Launching analysis for {selected_token}...")
    data = fetch_token_price(selected_token, days=180)
    name, ticker = selected_display.split(" (")
    ticker = ticker.rstrip(")")
    st.write(f"**Name:** {name} | **Ticker:** {ticker}")
    _, report = analyse_token(name, data, ticker)
    st.write(report)
    #path = "./models/tigerV2_20250807_152739.pt"
    path = f'./models/{selected_models}'
    from tigerV2 import run_model_and_plot
    try:
        Bbuff = run_model_and_plot(path, data)
        st.image(Bbuff)
    except Exception as e:
        st.write(e)
        st.error('Get limit rate, wait 60 sec before call an analyse !')