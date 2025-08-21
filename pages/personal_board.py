import streamlit as st

st.set_page_config(page_title="Personal Board", layout="wide", page_icon="👁️")
st.title("Personal Board page")


st.session_state['user'] = st.text_input("Username")

if st.session_state['user']:
    st.write(f"Welcome {st.session_state['user']}")