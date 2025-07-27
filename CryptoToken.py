class Token:
    def __init__(self, name, ticker, price, variation_24h, category):
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
