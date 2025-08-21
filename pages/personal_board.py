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

    for rubrick in board["rubricks"]:
        with st.expander(f"📂 {rubrick['name']}"):
            for item in rubrick["items"]:
                st.write(f"- **{item['symbol']}** | Buy: {item['buy_price']} | Qty: {item['quantity']}")
                if st.button(f"❌ Remove {item['symbol']}", key=f"rm_{item['symbol']}_{rubrick['name']}"):
                    delete_item(board_name, rubrick["name"], item["symbol"])
                    st.rerun()

            st.write("➕ Add Investment")
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol = st.text_input(f"Symbol ({rubrick['name']})", key=f"sym_{rubrick['name']}")
            with col2:
                buy_price = st.number_input("Buy Price", min_value=0.0, key=f"price_{rubrick['name']}")
            with col3:
                quantity = st.number_input("Quantity", min_value=0.0, key=f"qty_{rubrick['name']}")

            if st.button(f"Add {rubrick['name']} Investment", key=f"add_{rubrick['name']}"):
                if symbol:
                    add_item(board_name, rubrick["name"], symbol, buy_price, quantity)
                    st.rerun()

        if st.button(f"🗑️ Delete Rubrick {rubrick['name']}", key=f"del_{rubrick['name']}"):
            delete_rubrick(board_name, rubrick["name"])
            st.rerun()

    st.write("---")
    new_rubrick = st.text_input("➕ Add new rubrick")
    if st.button("Add Rubrick") and new_rubrick:
        add_rubrick(board_name, new_rubrick)
        st.rerun()
