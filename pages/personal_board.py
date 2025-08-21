import streamlit as st

st.set_page_config(page_title="Personal Board", layout="wide", page_icon="👁️")
st.title("Personal Board page")

# name = st.session_state['user']
#
# print(st.session_state['user'])
if st.button("Predict"):
    print(st.session_state)
#st.write(f"{name}")
