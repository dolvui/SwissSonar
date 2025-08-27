import streamlit as st

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='select which action need to make')
    parser.add_argument('--refresh-coins', metavar='boolean', required=False)
    parser.add_argument('--train', metavar='int', required=False)
    parser.add_argument('--tickers', metavar='string', required=False)

    args = parser.parse_args()
    #from sqliteModels import init_db
    #init_db()
    if args.refresh_coins:
        from scripts.daily_refresh import refresh_coins
        refresh_coins()
    if args.train:
        from scripts.model_trainner import launch_train_model
        launch_train_model(args.train,args.tickers)
    if not args.train and not args.refresh_coins:
        pages = {
            "Home": [
                st.Page("pages/home.py", title="Home page"),
                st.Page("pages/personal_board.py", title="Personal board"),
            ],
            "Resources": [
                st.Page("pages/crypto_page.py", title="Crypto tools"),
                st.Page("pages/train_model.py", title="model Crypto"),
            ],
        }
        pg = st.navigation(pages,position="top")
        pg.run()