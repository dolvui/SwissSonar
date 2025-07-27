from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import re
from CryptoToken import Token


options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.get("https://swissborg.com/fr/supported-assets")
time.sleep(3)

def get_swissUpadte() -> ([Token],[str]) :
    tokens : [Token] = []
    new_tokens : [str] = []

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    tbody = table.find("tbody") if table else None
    rows = tbody.find_all("tr") if tbody else []

    for row in rows:
        data = row.find_all("p")
        token = Token(data[0].text,data[1].text,data[2].text,data[3].text,data[4].text)
        tokens.append(token)

    names = soup.find_all('p',{'class':'row-title'})
    for name in names:
        parent_new = name.find_parent('div',{'class': 'assetsCardsSlice__SCardsContainer-sc-c2obde-2'})
        if parent_new:
            new_tokens.append(name.text)
    return tokens,new_tokens