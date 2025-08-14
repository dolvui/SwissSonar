from old_files.sqliteDB import fetch_last_7_days

def detect_anomalies(token_name):
    history = fetch_last_7_days(token_name)
    if not history or len(history) < 2:
        return None

    last = history[-1]

    reddit_vals = [day['reddit_mentions'] for day in history[:-1] if day['reddit_mentions'] != -1]
    trend_vals = [day['trend_score'] for day in history[:-1] if day['trend_score'] != -1]
    yt_vals = [day['youtube_mentions'] for day in history[:-1] if day['youtube_mentions'] != -1]

    avg_reddit = sum(reddit_vals) / len(reddit_vals) if reddit_vals else 0
    avg_trend = sum(trend_vals) / len(trend_vals) if trend_vals else 0
    avg_yt = sum(yt_vals) / len(yt_vals) if yt_vals else 0

    score = (avg_trend * 2) + (avg_reddit) + (avg_yt)

    anomalies = {}

    if avg_reddit > 0 and last['reddit_mentions'] > avg_reddit * 2:
        anomalies['reddit'] = (last['reddit_mentions'], avg_reddit)

    if avg_trend > 0 and last['trend_score'] > avg_trend * 2:
        anomalies['google_trends'] = (last['trend_score'], avg_trend)

    if anomalies:
        check_and_alert(token_name, anomalies)

    return score

def check_and_alert(token_name, anomalies):
    msg = f"🚨 Anomalie détectée sur {token_name} :\n\n"
    for source, (current, avg) in anomalies.items():
        msg += f"🔹 {source} : {current} (moyenne : {avg:.1f})\n"

    print(msg)
    #send_alert(f"🚨 ALERTE Crypto : {token_name}", msg)
