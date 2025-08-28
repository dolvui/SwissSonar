from board import get_all_boards
from analysis_preliminaire import analyse_token
from coingeckoAPI import fetch_token_price
import smtplib
from email.mime.text import MIMEText
import os

def send_mail(global_report):
    to_email = os.environ["MAIL_TO"]
    smtp_user = os.environ["MAIL_FROM"]
    smtp_pass = os.environ["MAIL_PASS"]

    msg = MIMEText(global_report)
    msg["Subject"] = "SwissSonar – Crypto Board Alert"
    msg["From"] = smtp_user
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())

def analyse_board():
    boards = get_all_boards()
    results = []
    reports = []
    global_report = ""
    for board in boards:
        for rubrick in board["rubricks"]:
            if rubrick["provider"] == "crypto":
                for item in rubrick["items"]:
                    ticker = item["symbol"]
                    cg_data = fetch_token_price(ticker)

                    # handle rate limit
                    if "prices" not in cg_data:
                        from time import sleep
                        sleep(60)
                        cg_data = fetch_token_price(ticker)

                    obj, report = analyse_token("", cg_data, item["symbol"])
                    results.append(obj)
                    reports.append(report)

    global_report = "\n".join(reports)
    peak_alerts = []
    for obj in results:
        if obj.get("signal") in ["🚨 Exceptional", "🔺 Strong"]:
            peak_alerts.append(f"🚀 {obj['ticker']} looks hot! Score={obj['signal_score']}")

    send_mail(global_report)