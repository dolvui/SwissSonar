class Token:
    def __init__(self, name, ticker, price, variation_24h, category):
        self.id = None,
        self.name = name
        self.ticker = ticker
        self.price = price.replace('$','')
        self.variation_24h = variation_24h
        self.category = category
        self.current_price = None
        self.market_cap = None
        self.volume_24h = None
        self.change_24h = None
        self.is_new = False
        self.trend_score = 0
        self.reddit_mentions = 0
        self.youtube_mentions = 0

    def to_dict(self):
        return {
            "name": self.name,
            "ticker": self.ticker,
            "price": self.price,
            "variation_24h": self.variation_24h,
            "category": self.category
        }

    def __repr__(self):
        return f"{self.name} ({self.ticker}) - ${self.price} - {self.variation_24h}"

    def dict_data(self):
        return {
        "id" : self.id,
        "name" : self.name,
        "ticker" : self.ticker,
        "price": self.price,
        "variation_24h" : self.variation_24h ,
        "category" : self.category,
        "current_price": self.current_price,
        "market_cap" : self.market_cap,
        "volume_24h" : self.volume_24h,
        "change_24h" : self.change_24h,
        "is_new" : self.is_new,
        "trend_score" : self.trend_score,
        "reddit_mentions" : self.reddit_mentions,
        "youtube_mentions" : self.youtube_mentions
        }

    def string_data(self):
        return f" {self.name} ({self.ticker}) : \n \
        price: {self.price} \n \
        variation_24h : {self.variation_24h} \n \
        category : {self.category} \n \
        current_price: {self.current_price} \n \
        market_cap : {self.market_cap} \n \
        volume_24h : {self.volume_24h} \n \
        change_24h : {self.change_24h} \n \
        is_new : {self.is_new} \n \
        trend_score : {self.trend_score} \n \
        reddit_mentions : {self.reddit_mentions} \n \
        youtube_mentions : {self.youtube_mentions} \n "

def entity_to_token(dict):
    token = Token(dict["name"],dict["ticker"],"",0,dict["category"])
    token.id = dict["gecko_id"]
    token.name = dict["name"]
    token.ticker = dict["ticker"]
    token.price = f"${dict["current_price"]}"
    #token.variation_24h = dict["variation_24h"]
    token.category = dict["category"]
    token.current_price = dict["current_price"]
    token.market_cap = dict["market_cap"]
    token.volume_24h = dict["volume_24h"]
    token.change_24h = dict["change_24h"]
    token.is_new = dict["is_new"]
    token.trend_score = dict["trend_score"]
    token.reddit_mentions = dict["reddit_mentions"]
    token.youtube_mentions = dict["youtube_mentions"]
    return token