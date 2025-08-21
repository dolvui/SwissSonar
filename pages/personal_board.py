import streamlit as st
from board import get_board, add_rubrick, delete_rubrick, add_item, delete_item

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
    st.title(f"📊 Board: {board_name}")

    board = get_board(board_name)
    total_board_pnl = 0.0

    for rubrick in board["rubricks"]:
        rubrick_pnl = 0.0
        with st.expander(f"📂 {rubrick['name']} ({rubrick.get('provider','?')})"):
            # Header row
            col1, col2, col3, col4, col5 = st.columns([2,2,2,2,1])
            col1.write("**Symbol**")
            col2.write("**Buy Price**")
            col3.write("**Qty**")
            col4.write("**Current**")
            col5.write("**Δ%**")

            for item in rubrick["items"]:
                current_price = 42#get_price(rubrick.get("provider"), item["symbol"])
                delta = (current_price - item["buy_price"]) / item["buy_price"] * 100 if item["buy_price"] > 0 else 0
                rubrick_pnl += (current_price - item["buy_price"]) * item["quantity"]

                col1, col2, col3, col4, col5 = st.columns([2,2,2,2,1])
                col1.write(f"**{item['symbol']}**")
                col2.write(item["buy_price"])
                col3.write(item["quantity"])
                col4.write(current_price)
                col5.markdown(
                    f"<span style='color: {'green' if delta >= 0 else 'red'}'>{delta:+.2f}%</span>",
                    unsafe_allow_html=True
                )

                if st.button(f"❌", key=f"rm_{item['symbol']}_{rubrick['name']}"):
                    delete_item(board_name, rubrick["name"], item["symbol"])
                    st.rerun()

            st.info(f"Subtotal {rubrick['name']}: {rubrick_pnl:+.2f}")
            total_board_pnl += rubrick_pnl

        if st.button(f"🗑️ Delete Rubrick {rubrick['name']}", key=f"del_{rubrick['name']}"):
            delete_rubrick(board_name, rubrick["name"])
            st.rerun()

    st.success(f"🏦 Total P&L for Board: {total_board_pnl:+.2f}")

    st.write("---")
    new_rubrick = st.text_input("➕ Add new rubrick")
    provider = st.selectbox("Provider", ["crypto", "stock", "forex"], key="provider")
    if st.button("Add Rubrick") and new_rubrick:
        add_rubrick(board_name, {"name": new_rubrick, "provider": provider})
        st.rerun()
