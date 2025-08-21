import praw
from googleapiclient.discovery import build
import json
from datetime import datetime, timedelta
from pytrends.request import TrendReq as PyTrendReq
import pandas as pd
from CryptoToken import Token
import streamlit as st

pd.set_option('future.no_silent_downcasting', True)


headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'fr-FR,fr;q=0.7',
    'cache-control': 'max-age=0',
    'priority': 'u=0, i',
    'referer': 'https://trends.google.com/',
    'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Brave";v="138"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version-list': '"Not)A;Brand";v="8.0.0.0", "Chromium";v="138.0.0.0", "Brave";v="138.0.0.0"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-ua-platform-version': '"10.0.0"',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'sec-gpc': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    # 'cookie': 'HSID=AqoCrIyWeamAxsRRD; SSID=A6fb5OXQ7qoNRMzVF; APISID=y4Lh01nEBM8rA0uR/AD5QY4wAjWuGuYhpt; SAPISID=U-zhxNGx6NUpOBoP/A-lu63RaeHJ1plKMl; __Secure-1PAPISID=U-zhxNGx6NUpOBoP/A-lu63RaeHJ1plKMl; __Secure-3PAPISID=U-zhxNGx6NUpOBoP/A-lu63RaeHJ1plKMl; SID=g.a000ygj4z5z7DvWhYmtN03iOJS1B9pp39nZgCdCWj6VitU6gC39qb1qOcgsEEKKizGxyJXLtvwACgYKAbASARESFQHGX2MiYzU7dESH5FModga63Mr5SRoVAUF8yKpZSgICj3fLYx8Hm2n0yynK0076; __Secure-1PSID=g.a000ygj4z5z7DvWhYmtN03iOJS1B9pp39nZgCdCWj6VitU6gC39qC9Og4mxnbLlL70omY9CcWgACgYKAcoSARESFQHGX2MiBZ1QSuBbrLAJ-sZMkf-uyRoVAUF8yKqRHg3PPS73OMWZxpcyIQ100076; __Secure-3PSID=g.a000ygj4z5z7DvWhYmtN03iOJS1B9pp39nZgCdCWj6VitU6gC39q5-U66J4UR_NLkZRR83mq0gACgYKAcoSARESFQHGX2MiVb5Vr7OZ31wgOhFC1aCx8hoVAUF8yKq7ico35d9mIsafp1d1PmMZ0076; SEARCH_SAMESITE=CgQIu54B; S=billing-ui-v3=zcK2vkKPW7HHq-uNg91Z9SzgMkrklcQ6:billing-ui-v3-efe=zcK2vkKPW7HHq-uNg91Z9SzgMkrklcQ6; AEC=AVh_V2gZj5mOsbjbkszSXykRZESNKAIzWag8ebC5DHZj3D5FhHaDjAPAIiI; __Secure-ENID=28.SE=cqIk87j4Jk3QF4HRVqW5sk6qn4__L58icTWvIsNeCFhLgmS7O9M4EkjnQNpIXuUWC9ieiOYIilD_3fKe7laq-173VIGjcKwLGP5aTw5NfRx4r1jcbMOwK7iSG0AG7eARYC3Xj9T0AqlHEzPMAxa6wH5qsdFzSMmdWA92pRo4un8PqCdrisz6ztCrZ-IQwXTcFxbAP-2gL0O2-VH3iIklVboNeh6oSMc9qFa6P6KrIao6rq9lfaFwB4vM95I0AD1CXzeehiTPgYAc2CgxG2MXRbG6s0Q10mc5vEjg7MAHwC1Ef8IU1GFAcGeMtpyLp57ah3oUNvwFx74oLoJvbydreFlKijz8l28HUvzfDFP44ZHhPTMJUvLMXwrZCU2MiKUo; NID=525=EoJA1gqi4ihR-xntayDAFNUjxN-yU9jlQbcCdtH_uMQiB9lzOQi1KSGQ_qeT5h64_YDxNgQ0UuXWV3NpTE7glkvxZf0c4GHc4Can-8LjZ_rOCd9XIraGHkVQXbGK3E2dQEbNiPB8Lxs-7G9hDJipNOAUA8eVWKWs4or7Ez-ouPNYuOofjQA2JresDbJFXJvhdhXBuSKjC84e2LCwrnQFTfo8naAqeZ02wiMvaRlxMLqmAJ--6BZ3leSc9wEf901FL1lR40zgyIwpblt8jlZG_xd0Z-zjwAHt5N4wx8I04sjcNxP6vqm_KpcnIVNgOJoPPN9oSYZrUnzNpQPJqqMQHgNGSHl6Li28X0B_HKZyTZsINsI0Vkrj__pQrwutkQn5oB1nEqLCBU7RQQ; __Secure-1PSIDTS=sidts-CjEB5H03P266FNV1WHoRCMe00e8KqeCJbbQeUenPdU_t97ue9IW-Ui6pJ4r0CZBL58NmEAA; __Secure-3PSIDTS=sidts-CjEB5H03P266FNV1WHoRCMe00e8KqeCJbbQeUenPdU_t97ue9IW-Ui6pJ4r0CZBL58NmEAA; OTZ=8184710_48_52_123900_48_436380; SIDCC=AKEyXzXXjWnbUnaPvUOvhH524XUln0G2UxSA-5Z86c5KrYBvnsXk-PPSHRzbSmpbpjikp9sYzw; __Secure-1PSIDCC=AKEyXzXaOejkQV4_zZmiVpljcyTlFu2xsioQZF85OBjUMJ9NpDD0EE7HDLrlh7id4gonilSyMQ; __Secure-3PSIDCC=AKEyXzWHDdI2PPIH8Tp5yNhdl3eS0bx-25Vrl2PSUfIY0TaC60BrpbpAxtM35p6A_lh6uPv_h-I',
}

class TrendReq(PyTrendReq):
    def _get_data(self, url, method='get', trim_chars=0, **kwargs):
        return super()._get_data(url, method=method, trim_chars=trim_chars, headers=headers, **kwargs)


secret = None
reddit = None
key = None
is_running_on_github = True

try:
    with open("secret.json", 'r', encoding='utf-8') as json_file:
        secret = json.load(json_file)
except:
    try:
        secret = st.secrets
        is_running_on_github = False
    except:
        is_running_on_github = False

if is_running_on_github:
    import os

    reddit = praw.Reddit(
        client_id=os.environ['CLIENT_ID'],
        client_secret=os.environ['CLIENT_SECRET'],
        user_agent=os.environ['USER_AGENT']
    )
    key = os.environ['GOOGLE_KEY']
else:
    reddit = praw.Reddit(
        client_id=secret['client_id'],
        client_secret=secret['client_secret'],
        user_agent=secret['user_agent']
    )
    key = secret['google_key']





pytrends = TrendReq(hl='en-US', tz=360)


def fetch_online_trend(tokens : [Token]):
    rep = []
    for token in tokens:
        print(f"Processing {token.name}")
        token.trend_score = get_google_trend_score(token.name)
        token.reddit_mentions = count_reddit_mentions(token.name)
        token.youtube_mentions = youtube_search_count(token.name)
        rep.append(token)
    return rep


def get_google_trend_score(keyword):
    try:
        kw_list = [keyword]
        pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')

        data = pytrends.interest_over_time()
        print(data)
        return int(data[keyword].iloc[-1])
    except Exception as e:
        print("google Failed ! : \n",e)
        return -1

def youtube_search_count(query, api_key=key, max_results=1000):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)

        published_after = (datetime.now() - timedelta(days=1)).isoformat("T") + "Z"

        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            publishedAfter=published_after
        )
        response = request.execute()
        return len(response.get("items", []))
    except Exception as error:
        print("youtube Failed ! : \n",error)
        return -1

def count_reddit_mentions(keyword, subreddit="cryptocurrency", limit=1000):
    count = 0
    try:
        for _ in reddit.subreddit(subreddit).search(keyword, limit=limit):
            count += 1
    except Exception as e:
        print("reddit Failed ! : \n", e)
        return -1
    return count

def compute_heuristics(google_trend, youtube_mentions, reddit_mentions, previous_google, previous_youtube, previous_reddit):
    delta_google = ((google_trend - previous_google) / 200)
    delta_youtube = ((youtube_mentions - previous_youtube) / 2000)
    delta_reddit = ((reddit_mentions - previous_reddit) / 2000)

    yt = (int(youtube_mentions != -1 and previous_youtube != -1) * 2) + delta_youtube
    gg = (int(google_trend != -1 and previous_google != -1) * 5) + delta_google
    rd = (int(reddit_mentions != -1 and previous_reddit != -1) * 3) + delta_reddit

    h = (( gg*(google_trend/100) + yt*(youtube_mentions/1000) + rd*(reddit_mentions/1000))*10)/ (yt + gg + rd)
    return h