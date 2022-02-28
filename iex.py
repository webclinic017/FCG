import requests
import datetime
from dateutil.relativedelta import relativedelta

class IEXStock:

    def __init__(self, token, symbol, environment='production'):
        if environment == 'production':
            self.BASE_URL = 'https://cloud.iexapis.com/v1'
        else:
            self.BASE_URL = 'https://sandbox.iexapis.com/v1'
        
        self.token = token
        self.symbol = symbol

    def get_logo(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/logo?token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_company_info(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/company?token={self.token}"
        r = requests.get(url)

        return r.json()
    
    def get_company_news(self, st, last=10):
        url = f"{self.BASE_URL[:-3]}/stable/stock/{self.symbol}/news/last/50?token={self.token}"
        # to show working api call st.text(url)
        r = requests.get(url)

        return r.json()

    def get_recommendations(self, st, last=10):
        url = f"{self.BASE_URL[:-3]}/stable/time-series/CORE_ESTIMATES/{self.symbol}?token={self.token}"
        # to show working api call
        r = requests.get(url)
        return r.json()



    def get_stats(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/advanced-stats?token={self.token}"
        r = requests.get(url)
        
        return r.json()

    def get_fundamentals(self, st, period='quarterly', last=4):
        url = f"{self.BASE_URL}/time-series/fundamentals/{self.symbol}/{period}?last={last}&token={self.token}"
        r = requests.get(url)
        return r.json()

    def get_dividends(self, range='5y'):
        url = f"{self.BASE_URL}/stock/{self.symbol}/dividends/{range}?token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_institutional_ownership(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/institutional-ownership?token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_insider_transactions(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/insider-transactions?token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_oil_prices(self,st,  range='10y'):
        url = f"{self.BASE_URL[:-3]}/stable/time-series/energy/DCOILWTICO?range={range}&token={self.token}"
        #st.text(url)
        r = requests.get(url)

        return r.json()

    def get_fx_currency(self,cur_1,cur_2,amount):
        url = f"{self.BASE_URL[:-3]}/stable/fx/convert?symbols={cur_1}{cur_2}&amount={amount}&token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_sector_prices(self,st,  range='10y'):
        url = f"{self.BASE_URL[:-3]}/stable/stock/market/sector-performance?token={self.token}"
        r = requests.get(url)

        return r.json()


    def get_cpi_prices(self, st, range='5'):
        new_date = (datetime.datetime.today() - relativedelta(years=int(range))).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL[:-3]}/stable/time-series/ECONOMIC/CPIAUCSL?from={new_date}&token={self.token}"

        #st.text(url)
        r = requests.get(url)

        return r.json()

    def get_peer_group(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/peers?token={self.token}"
        r = requests.get(url)



