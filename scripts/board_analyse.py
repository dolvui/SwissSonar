from board import get_all_boards
from analysis_preliminaire import analyse_token
from coingeckoAPI import fetch_token_price
import smtplib
from email.mime.text import MIMEText
import os
from mongodb import fetch_token_24h
from prices import get_price_cryptocurrency

def send_mail(global_report):
    to_email = os.environ["MAIL_TO"]
    smtp_user = os.environ["MAIL_FROM"]
    smtp_pass = os.environ["MAIL_PASS"]

    msg = MIMEText(global_report)
    msg["Subject"] = "SwissSonar – Crypto Board Alert"
    msg["From"] = smtp_user
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())

def compute_pnl(board, cryptos_available):
    rubrick_results = []
    total_board_pnl = 0.0

    for rubrick in board["rubricks"]:
        rubrick_pnl = 0.0
        rubrick_items = []

        if rubrick.get("provider") == "crypto":
            for item in rubrick["items"]:
                current_price = get_price_cryptocurrency(
                    cryptos_available[item["symbol"]],
                    item["buy_price"]
                )
                delta = (current_price - item["buy_price"]) / item["buy_price"] * 100 if item["buy_price"] > 0 else 0
                pnl_value = (current_price - item["buy_price"]) * item["quantity"]
                rubrick_pnl += pnl_value

                rubrick_items.append({
                    "symbol": item["symbol"],
                    "buy_price": item["buy_price"],
                    "quantity": item["quantity"],
                    "current": current_price,
                    "delta": delta,
                    "pnl_value": pnl_value,
                })

            total_board_pnl += rubrick_pnl
            rubrick_results.append((rubrick["name"], rubrick_pnl, rubrick_items))

    return total_board_pnl, rubrick_results

def analyse_market(symbol, gecko_id):
    cg_data = fetch_token_price(gecko_id)

    if "prices" not in cg_data:
        from time import sleep
        sleep(60)  # Rate-limit protection
        cg_data = fetch_token_price(gecko_id)

    obj, report = analyse_token(symbol, cg_data, gecko_id)
    return obj, report

def build_reports(boards, cryptos_available):
    portfolio_summary = []
    analysis_reports = []
    peak_alerts = []

    for board in boards:
        total_pnl, rubrick_results = compute_pnl(board, cryptos_available)
        portfolio_summary.append(f"Board {board['board_name']} PnL: {round(total_pnl, 2)} USD")

        for rubrick_name, rubrick_pnl, items in rubrick_results:
            for item in items:
                symbol = item["symbol"]
                gecko_id = cryptos_available[symbol]

                obj, report = analyse_market(symbol, gecko_id)

                # Enrich report with portfolio delta
                report = f"{report}\nPnL Delta: {round(item['delta'], 2)}%\n"
                analysis_reports.append(report)

                # Collect alerts
                if obj.get("signal") in ["🚨 Exceptional", "🔺 Strong"]:
                    peak_alerts.append(f"🚀 {obj['ticker']} ({symbol}) looks hot! Score={obj['signal_score']}")

    global_report = "\n".join(portfolio_summary + analysis_reports)
    return global_report, peak_alerts

def analyse_board():
    boards = get_all_boards()
    cryptos_available = {e['ticker']: e['gecko_id'] for e in fetch_token_24h()}

    global_report, peak_alerts = build_reports(boards, cryptos_available)

    # Send daily digest
    send_mail(global_report)

    # Extra instant alert (optional)
    if peak_alerts:
        send_mail("\n".join(peak_alerts))


# def analyse_board():
#     boards = get_all_boards()
#     results = []
#     reports = []
#
#     cryptos_available = {e['ticker']: e['gecko_id'] for e in fetch_token_24h()}
#
#
#     for board in boards:
#         rubrick_results = []
#         total_board_pnl = 0.0
#         for rubrick in board["rubricks"]:
#             rubrick_pnl = 0.0
#             rubrick_items = []
#             if "provider" in rubrick and rubrick["provider"] == "crypto":
#                 for item in rubrick["items"]:
#                     current_price = 0.0
#
#                     if rubrick["provider"] == "crypto":
#                         current_price = get_price_cryptocurrency(cryptos_available[item["symbol"]], item["buy_price"])
#                     delta = (current_price - item["buy_price"]) / item["buy_price"] * 100 if item["buy_price"] > 0 else 0
#                     pnl_value = (current_price - item["buy_price"]) * item["quantity"]
#                     rubrick_pnl += pnl_value
#
#                     rubrick_items.append({
#                         "symbol": item["symbol"],
#                         "buy_price": item["buy_price"],
#                         "quantity": item["quantity"],
#                         "current": current_price,
#                         "delta": delta,
#                     })
#
#                 total_board_pnl += rubrick_pnl
#                 rubrick_results.append((rubrick, rubrick_pnl, rubrick_items))
#                 ticker = item["symbol"]
#                 id = cryptos_available[item["symbol"]]
#
#                 cg_data = fetch_token_price(id)
#
#                 # handle rate limit
#                 if "prices" not in cg_data:
#                     from time import sleep
#                     sleep(60)
#                     cg_data = fetch_token_price(id)
#
#                 obj, report = analyse_token(ticker, cg_data, id)
#                 report = f"\n{report}Delta : {item['delta']} \n"
#                 results.append(obj)
#                 reports.append(report) # send
#
#     global_report = "\n".join(reports)
#     # i dont like this logic , also the fact that there is a analyse there retrurn the obj :
#     # peak = detect_peak(prices)
#     #
#     # obj = {
#     #     "name": name,
#     #     "ticker": ticker,
#     #     "delta_price_pct": round(delta_price, 2),
#     #     "volatility_volume": round(vol_volatility, 2),
#     #     "corr_price_volume": round(corr_pv, 2),
#     #     "slope": round(slope, 2),
#     #     "r2": round(r2, 2),
#     #     "signal_score": round(score, 2),
#     #     "signal": signal,
#     #     "comment": comment,
#     #     "peak": peak
#     # }
#     #peak_alerts = []
#     # for obj in results:
#     #     if obj.get("signal") in ["🚨 Exceptional", "🔺 Strong"]:
#     #         peak_alerts.append(f"🚀 {obj['ticker']} looks hot! Score={obj['signal_score']}")
#
#     # also send the email if force to or the tendency change , up but moslty down
#     send_mail(global_report)