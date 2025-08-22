import streamlit as st
from board import get_board, add_rubrick, delete_rubrick, add_item, delete_item
from prices import get_price_cryptocurrency, get_price_stock, get_price_forex
from mongodb import fetch_token_24h
from pytickersymbols import PyTickerSymbols
stock_data = PyTickerSymbols()

cryptos_available = {e['ticker']: e['gecko_id'] for e in fetch_token_24h()}
stock_symbols = PyTickerSymbols().get_all_stocks() #get_stock_by_google_symbol(None)#
forex_available = ["USD", "EUR", "GBP", "JPY"]

st.set_page_config(page_title="Personal Board", layout="wide", page_icon="👁️")

if "board_name" not in st.session_state:
    st.session_state["board_name"] = None

if st.session_state["board_name"] is None:
    st.title("📋 Select Your Investment Board")
    board_name = st.text_input("Enter a board name:")
    if st.button("Load Board") and board_name:
        st.session_state["board_name"] = board_name
        st.rerun()
else:
    board_name = st.session_state["board_name"]
    #st.title(f"📊 Board: {board_name}")

    board = get_board(board_name)

    # --- compute all pnl before rendering ---
    rubrick_results = []
    total_board_pnl = 0.0

    for rubrick in board["rubricks"]:
        rubrick_pnl = 0.0
        rubrick_items = []

        for item in rubrick["items"]:
            current_price = 0.0

            if rubrick["provider"] == "crypto":
                current_price = get_price_cryptocurrency(cryptos_available[item["symbol"]])
            if rubrick["provider"] == "stock":
                current_price = get_price_stock(item["symbol"])
            if rubrick["provider"] == "forex":
                current_price = get_price_forex(item["symbol"])
            delta = (current_price - item["buy_price"]) / item["buy_price"] * 100 if item["buy_price"] > 0 else 0
            pnl_value = (current_price - item["buy_price"]) * item["quantity"]
            rubrick_pnl += pnl_value

            rubrick_items.append({
                "symbol": item["symbol"],
                "buy_price": item["buy_price"],
                "quantity": item["quantity"],
                "current": current_price,
                "delta": delta,
            })

        total_board_pnl += rubrick_pnl
        rubrick_results.append((rubrick, rubrick_pnl, rubrick_items))

    # --- render board title ---
    board_color = "green" if total_board_pnl >= 0 else "red"
    st.markdown(
        f"<h1>📊 {board_name} <span style='float:right; color:{board_color};'>{total_board_pnl:+.2f}</span></h1>",
        unsafe_allow_html=True
    )

    # --- render rubricks ---
    for rubrick, rubrick_pnl, rubrick_items in rubrick_results:
        rubrick_color = "green" if rubrick_pnl >= 0 else "red"

        with st.expander(f"📂 {rubrick['name']} ({rubrick.get('provider', '?')} )       :{rubrick_color}[{rubrick_pnl:+.2f}]", expanded=False):
            # Header row
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            col1.write("**Symbol**")
            col2.write("**Buy Price**")
            col3.write("**Qty**")
            col4.write("**Current**")
            col5.write("**Δ%**")

            # Rows
            for item in rubrick_items:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                col1.write(f"**{item['symbol']}**")
                col2.write(item["buy_price"])
                col3.write(item["quantity"])
                col4.write(item["current"])
                col5.markdown(
                    f"<span style='color: {'green' if item['delta'] >= 0 else 'red'}'>{item['delta']:+.2f}%</span>",
                    unsafe_allow_html=True
                )

                if st.button(f"❌", key=f"rm_{item['symbol']}_{rubrick['name']}"):
                    delete_item(board_name, rubrick["name"], item["symbol"])
                    st.rerun()
            st.write("➕ Add Investment")
            col1, col2, col3 = st.columns(3)
            with col1:
                if rubrick["provider"] == "crypto":
                    symbol = st.selectbox("Select crypto available", options=cryptos_available.keys(),key=f"sym_{rubrick['name']}")
                if rubrick["provider"] == "stock":
                    print(stock_symbols)
                    symbol = st.selectbox("Select stock available", options=[f"{s['symbol']} - {s['name']}" for s in stock_symbols],key=f"sym_{rubrick['name']}")
                if rubrick["provider"] == "forex":
                    symbol = st.selectbox("Select forex available", options=forex_available,key=f"sym_{rubrick['name']}")
            with col2:
                buy_price = st.number_input("Buy Price", min_value=0.0, key=f"price_{rubrick['name']}")
            with col3:
                quantity = st.number_input("Quantity", min_value=0.0, key=f"qty_{rubrick['name']}")
            if st.button(f"Add {rubrick['name']} Investment", key=f"add_{rubrick['name']}"):
                if symbol:
                    add_item(board_name, rubrick["name"], symbol, buy_price, quantity)
                    st.rerun()
            total_board_pnl += rubrick_pnl

        if st.button(f"🗑️ Delete Rubrick {rubrick['name']}", key=f"del_{rubrick['name']}"):
            delete_rubrick(board_name, rubrick["name"])
            st.rerun()

    st.write("---")
    new_rubrick = st.text_input("➕ Add new rubrick")
    provider = st.selectbox("Provider", ["crypto", "stock", "forex"], key="provider")
    if st.button("Add Rubrick") and new_rubrick:
        add_rubrick(board_name, {"name": new_rubrick, "provider": provider})
        st.rerun()

st.write("---")
st.write("\n")
if board_name:
    st.write("change board")
    board_name = st.text_input("Enter a board name:")
    if st.button("Load Board"):
        st.session_state["board_name"] = board_name
        st.rerun()