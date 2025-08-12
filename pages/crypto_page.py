import pandas as pd
import streamlit as st
import coingeckoAPI
import swissUpdate
from CryptoToken import entity_to_token
from coingeckoAPI import fetch_token_price
from mongodb import fetch_token_24h
from mongodb import upsert_tokens_entry
from onlineTrend import fetch_online_trend

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
st.set_page_config(page_title="Crypto Analysis Dashboard", layout="wide")

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
        pass


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tokens", len(df_tokens))
with col2:
    top_marketcap = df_tokens.loc[df_tokens["market_cap"].idxmax()]["name"]
    st.metric("Highest Market Cap", top_marketcap)
with col3:
    top_trend = df_tokens.loc[df_tokens["trend_score"].idxmax()]["name"]
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
    filtered_df[["id","name", "ticker", "current_price", "volume_24h", "market_cap", "trend_score"]],
    use_container_width=True
)

# Select a token
selected_token = st.selectbox("Select a token for analysis", filtered_df["id"].tolist())

# Analyse button
if st.button("🔎 Analyse"):
    st.success(f"Launching analysis for {selected_token}...")
    data = fetch_token_price(selected_token, days=180)
    path = "./models/tigerV2_20250807_152739.pt"
    from tigerV2 import run_model_and_plot
    try:
        Bbuff = run_model_and_plot(path, data)
        st.image(Bbuff)
    except:
        st.error('Get limit rate, wait 60 sec before call an analyse !')