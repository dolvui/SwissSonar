from sqliteDB import fetch_last_7_days
#from alert import send_alert

def detect_anomalies(token_name):
    history = fetch_last_7_days(token_name)
    if not history or len(history) < 2:
        return None

    last = history[-1]
    avg_reddit = sum(day['reddit_mentions'] for day in history[:-1]) / (len(history) - 1)
    avg_trend = sum(day['trend_score'] for day in history[:-1]) / (len(history) - 1)

    anomalies = {}

    if avg_reddit > 0 and last['reddit_mentions'] > avg_reddit * 2:
        anomalies['reddit'] = (last['reddit_mentions'], avg_reddit)

    if avg_trend > 0 and last['trend_score'] > avg_trend * 2:
        anomalies['google_trends'] = (last['trend_score'], avg_trend)

    return anomalies if anomalies else None

def check_and_alert(token_name, anomalies):
    msg = f"🚨 Anomalie détectée sur {token_name} :\n\n"
    for source, (current, avg) in anomalies.items():
        msg += f"🔹 {source} : {current} (moyenne : {avg:.1f})\n"

    print(msg)
    #send_alert(f"🚨 ALERTE Crypto : {token_name}", msg)
