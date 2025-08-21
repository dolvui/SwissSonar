from board import get_board, add_rubrick, delete_rubrick, add_item, delete_item
import streamlit as st

try:
    user = st.session_state['user']
    board = get_board(user)
except Exception as e:
    st.session_state['user'] = st.text_input("Username")
    user = st.session_state['user']
    board = get_board(user)


st.subheader("📊 Your Rubricks")

def page():
    for rubrick in board["rubricks"]:
        with st.expander(rubrick["name"]):
            for item in rubrick["items"]:
                st.write(f"- {item}")
                if st.button(f"❌ Remove {item}", key=f"remove_{item}_{rubrick['name']}"):
                    delete_item(user, rubrick["name"], item)
                    st.rerun()

            new_item = st.text_input(f"Add item to {rubrick['name']}", key=f"new_{rubrick['name']}")
            if st.button(f"➕ Add to {rubrick['name']}", key=f"add_{rubrick['name']}"):
                add_item(user, rubrick["name"], new_item)
                st.rerun()

        if st.button(f"🗑️ Delete rubrick {rubrick['name']}", key=f"delete_{rubrick['name']}"):
            delete_rubrick(user, rubrick["name"])
            st.rerun()

    st.write("---")
    new_rubrick = st.text_input("➕ Add new rubrick")
    if st.button("Add Rubrick"):
        add_rubrick(user, new_rubrick)
        st.rerun()


if user:
    st.write(f"Welcome {user} !")
    page()
else:
    st.session_state['user'] = st.text_input("Username")
    user = st.session_state['user']
    board = get_board(user)