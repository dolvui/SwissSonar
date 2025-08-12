import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Noa's Hub", page_icon="🌐", layout="wide")

# --- HEADER ---
st.title("Noa's Hub")
st.markdown(
    """
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/noa-ghidalia-1877012b6) 
    [![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:noa@ghidalia.fr)
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/dolvui)
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- CARDS SECTION ---
st.subheader("Projects & Topics")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 💹 Crypto")
    st.write("Analysis, tracking, and prediction tools for cryptocurrency markets.")
    if st.button("Open Crypto Tools", key="crypto"):
        st.switch_page("streamlit\\crypto_page.py")

with col2:
    st.markdown("### 🏴‍☠️ One Piece")
    st.write("Fun data and trivia exploration about the One Piece universe.")
    if st.button("Open One Piece", key="one_piece"):
        st.switch_page("pages/one_piece.py")

with col3:
    st.markdown("### 🖥 Server Status")
    st.write("Real-time monitoring of servers and infrastructure health.")
    if st.button("Open Server Status", key="server"):
        st.switch_page("pages/server_status.py")

st.markdown("---")

# --- FOOTER ---
footer_text = """
<div style='text-align: center; font-size: 0.9rem; color: gray; padding: 20px 0;'>
This project is designed to centralize all of my work and interests.  
I began coding at 15 and have never stopped—today, my focus is on tackling big, challenging projects.
</div>
"""
st.markdown(footer_text, unsafe_allow_html=True)
