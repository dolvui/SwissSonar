import coingeckoAPI
import swissUpdate
from onlineTrend import *
from graph import plot_top_mentions
from sqliteDB import init_db, insert_token
init_db()
#
# tokens, new_ids = swissUpdate.get_swissUpadte()
#
# enriched_tokens = coingeckoAPI.fetch_market_data_fast(tokens, new_ids)

# for token in enriched_tokens:
#     try :
#         token.trend_score = get_google_trend_score(token.name)
#     except :
#         print('google failed')
#     try :
#         token.reddit_mentions = count_reddit_mentions(token.name)
#     except :
#         print('reddit failed')
#     try :
#         token.youtube_mentions = youtube_search_count(token.name)
#     except :
#         print('youtube failed')
#     insert_token(token.dict_data())

plot_top_mentions()