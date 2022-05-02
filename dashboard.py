import streamlit as st
import requests, redis
import config, json
from iex import IEXStock
from helpers import *
from datetime import datetime, timedelta
import datetime as datetime_parent
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
#from PIL import Image
from streamlit_tags import st_tags, st_tags_sidebar
import quantstats as qs
import make_report_file
import graphs
import helpers
import quantstats_modified as qs_mod
import quantstats_modified.reports as qs_mod_reports

stocks = get_stock_list()
if not os.path.exists('qs_reports'):
    os.makedirs('qs_reports')

etf_mapping = {"SPY": "S&P500", "DIA":"Dow Jones", "IWM": "Russell 2000"}
st.sidebar.image("Logo cropped.png", use_column_width=True)
#st.set_page_config(initial_sidebar_state="expanded")
#symbol = st.sidebar.text_input("Symbol/ticker")
keywords = st_tags_sidebar(
        label='Insert Stock or ticker {press enter}',
        text='enter ticker to confirm stock',
        suggestions=stocks,
        maxtags=1,
        key='1')
currency_symbol_list=["GBP", "USD", "EUR"]
try:
    symbol=keywords[0].lower()

    stock = IEXStock(config.IEX_TOKEN, symbol)

    client = redis.Redis(host="localhost", port=6379)

    screen = st.sidebar.selectbox("View", ('Homepage',
                                           'Overview',
                                           'Fundamentals',
                                           'Analyst Recommendations and News',
                                           'Ownership',
                                           'Open and Close Performance',
                                           'Performance indicators',
                                           'Oil and CPI',
                                           'FX',
                                           'Sector',

                                                     ), index=0)
except:
    pass
    symbol = ""
    screen = "Homepage"
def validate_symbol(symbol, st):
    if symbol.upper() not in stocks:
        return False
    return True
#if screen == "Error":
#    st.text("Please provide correct ticker/symbol")
if screen == 'Homepage':

    st.image('Logo cropped.png')
    st.title(screen)


    if not validate_symbol(symbol, st):
        st.text("Please enter a symbol to get started")
    else:

        logo_cache_key = f"{symbol}_logo"
        cached_logo = client.get(logo_cache_key)
        st.title("__Finer Consulting Group__")




        st.subheader("Welcome to Finer Advisory Group. Want investment and sector data, fx, news and stock predictions?")

        st.subheader("Use the left-side bar and input your desired ticker.")

        st.subheader("Then click on the desired function drop to load the data.")

        st.subheader("If you find the application useful please consider joining the premium account.")



        st.form('my_form_identifier')
        st.color_picker('Pick a color')

        st.markdown("DISCLAIMER:The investments and services offered by us may not be suitable for all investors.")
        st.markdown( "if you have any doubts as to the merits of an investment, you should seek advice from an independent financial advisor.")
        st.markdown("__Finer Advisory Group__")
        st.time_input('Time entry')
        #img= image.open("Telcast.jpeg")
        #st.image(img)
if screen == 'Overview':
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)

    if cached_logo is not None:
        print("found logo in cache")
        logo = json.loads(cached_logo)
    else:
        print("getting logo from api, and then storing it in cache")
        logo = stock.get_logo()
        client.set(logo_cache_key, json.dumps(logo))
        client.expire(logo_cache_key, timedelta(hours=24))

    company_cache_key = f"{symbol}_company"
    cached_company_info = client.get(company_cache_key)

    if cached_company_info is not None:
        print("found company news in cache")
        company = json.loads(cached_company_info)
    else:
        print("getting company from api, and then storing it in cache")
        company = stock.get_company_info()
        client.set(company_cache_key, json.dumps(company))
        client.expire(company_cache_key, timedelta(hours=24))

    col1, col2 = st.beta_columns([1, 4])

    with col1:
        st.image(logo['url'])

    with col2:
        st.subheader(company['companyName'])
        st.write(company['industry'])
        st.subheader('Description')
        st.write(company['description'])
        st.subheader('CEO')
        st.write(company['CEO'])

if screen == 'Analyst Recommendations and News':
    news_cache_key = f"{symbol}_news"

    #news = client.get(news_cache_key)

    #if news is not None:
    #    news = json.loads(news)
    #    st.text("Not getting company news")
    #else:
    news = stock.get_company_news(st,last=50)
        #print("line 85")
        #client.set(news_cache_key, json.dumps(news))
    st.subheader("Analyst Recommendations")
    recommendations= stock.get_recommendations(st)[0]
    ignore_keys = ['date', 'updated', 'subkey', 'key', 'id']
    keys = [key for key in recommendations.keys() if key not in ignore_keys]
    def check_is_marketconsensus(key):
        if key =="marketConsensus":
            return "*"
        else:
            return ""
    items = [str(recommendations[key])+check_is_marketconsensus(key) for key in keys]


    import pandas as pd

    dataframe = pd.DataFrame({'keys': keys, 'items': items})
    st.table(dataframe)
    st.text(convert_rating(recommendations["marketConsensus"]))

    st.write("50 to 100 = Strong Buy")

    st.write("25 to 50 = Buy")

    st.write("-25 to 25 = Neutral / Hold")

    st.write("-25 to -50 = sell")

    st.write("-50 to -100 = Strong Sell")

    st.subheader("News")
    for article in news:
        st.subheader(article['headline'])
        dt = datetime.utcfromtimestamp(article['datetime'] / 1000).isoformat()
        st.write(f"Posted by {article['source']} at {dt}")
        st.write(article['url'])
        st.write(article['summary'])
        st.image(article['image'])


if screen == 'Fundamentals':
    stats_cache_key = f"{symbol}_stats"
    stats = client.get(stats_cache_key)

    if stats is None:
        stats = stock.get_stats()
        client.set(stats_cache_key, json.dumps(stats))
    else:
        stats = json.loads(stats)

    st.header('Ratios')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('P/E')
        st.write(stats['peRatio'])
        st.subheader('Forward P/E')
        st.write(stats['forwardPERatio'])
        st.subheader('PEG Ratio')
        st.write(stats['pegRatio'])
        st.subheader('Price to Sales')
        st.write(stats['priceToSales'])
        st.subheader('Price to Book')
        st.write(stats['priceToBook'])
        st.subheader('profitMargin')
        st.write(stats['profitMargin'])
        st.subheader('debtToEquity')
        st.write(stats['debtToEquity'])
    with col2:
        st.subheader('Revenue')
        st.write(format_number(stats['revenue']))
        st.subheader('Cash')
        st.write(format_number(stats['totalCash']))
        st.subheader('Debt')
        st.write(format_number(stats['currentDebt']))
        st.subheader('200 Day Moving Average')
        st.write(stats['day200MovingAvg'])
        st.subheader('50 Day Moving Average')
        st.write(stats['day50MovingAvg'])
        st.subheader('beta')
        st.write(stats['beta'])

    fundamentals_cache_key = f"{symbol}_fundamentals"
    fundamentals = client.get(fundamentals_cache_key)

    if fundamentals is None:
        fundamentals = stock.get_fundamentals('quarterly', st=st)
        client.set(fundamentals_cache_key, json.dumps(fundamentals))
    else:
        fundamentals = json.loads(fundamentals)

    for quarter in fundamentals:
        st.header(f"Q{quarter['fiscalQuarter']} {quarter['fiscalYear']}")
        st.subheader('Filing Date')
        st.write(quarter['filingDate'])
        st.subheader('Revenue')
        st.write(format_number(quarter['revenue']))
        st.subheader('Net Income')
        st.write(format_number(quarter['incomeNet']))

    st.header("Dividends")

    dividends_cache_key = f"{symbol}_dividends"
    dividends = client.get(dividends_cache_key)

    if dividends is None:
        dividends = stock.get_dividends()
        client.set(dividends_cache_key, json.dumps(dividends))
    else:
        dividends = json.loads(dividends)

    for dividend in dividends:
        st.write(dividend['paymentDate'])
        st.write(dividend['amount'])

if screen == 'Ownership':
    st.subheader("Institutional Ownership")

    institutional_ownership_cache_key = f"{symbol}_institutional"
    institutional_ownership = client.get(institutional_ownership_cache_key)

    if institutional_ownership is None:
        institutional_ownership = stock.get_institutional_ownership()
        client.set(institutional_ownership_cache_key, json.dumps(institutional_ownership))
    else:
        print("getting inst ownership from cache")
        institutional_ownership = json.loads(institutional_ownership)

    for institution in institutional_ownership:
        st.write(institution['date'])
        st.write(institution['entityProperName'])
        st.write("{:,}".format(institution['reportedHolding']))

    st.subheader("Insider Transactions")

    insider_transactions_cache_key = f"{symbol}_insider_transactions"

    insider_transactions = client.get(insider_transactions_cache_key)
    if insider_transactions is None:
        insider_transactions = stock.get_insider_transactions()
        client.set(insider_transactions_cache_key, json.dumps(insider_transactions))
    else:
        print("getting insider transactions from cache")
        insider_transactions = json.loads(insider_transactions)

    for transaction in insider_transactions:
        st.write(transaction['filingDate'])
        st.write(transaction['fullName'])
        st.write("{:,}".format(transaction['transactionShares']))
        st.write(transaction['transactionPrice'])


def get_dataframe(tickers, period='30d', start_date=None, end_date=datetime.now()):
    tickers_hist = {ticker: yf.Ticker(ticker).history(period=period)
                    for ticker in tickers}
    ticker = tickers[0]
    start_date += datetime_parent.timedelta(days=-1)
    end_date += datetime_parent.timedelta(days=+1)
    start_date_combined = datetime.combine(start_date, datetime.min.time())
    end_date_combined = datetime.combine(end_date, datetime.min.time())
    data = tickers_hist[ticker]
    datetimes = np.array([date for date in data.index])
    indices = np.logical_and(datetimes > start_date_combined, datetimes < end_date_combined)
    tickers_hist[ticker] = tickers_hist[ticker].loc[indices]
    data = tickers_hist[ticker]
    return data

def plot_tickers(tickers, period='30d', start_date=None, end_date=datetime.now()):
    tickers_hist = {ticker: yf.Ticker(ticker).history(period=period)
                    for ticker in tickers}
    ticker=tickers[0]
    start_date += datetime_parent.timedelta(days=-1)
    end_date += datetime_parent.timedelta(days=+1)
    start_date_combined = datetime.combine(start_date, datetime.min.time())
    end_date_combined = datetime.combine(end_date, datetime.min.time())
    data = tickers_hist[ticker]
    datetimes = np.array([date for date in data.index])
    indices = np.logical_and(datetimes > start_date_combined, datetimes < end_date_combined)
    tickers_hist[ticker] = tickers_hist[ticker].loc[indices]
    data = tickers_hist[ticker]
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name=ticker)])

    #fig.update_xaxes(type='category')
    fig.update_layout(height=700)
    fig.update_layout(width=400)

    st.plotly_chart(fig, use_container_width=True)

    st.write(data)
    for item in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', ]:

        for ticker in tickers:


            tickers_hist[ticker][item].plot(label=ticker)

            print(tickers_hist[ticker].head())

            fig = px.line(tickers_hist[ticker], x=tickers_hist[ticker].index, y=item, title=item)

                # fig.show()

            st.plotly_chart(fig)

            st.download_button('Download', convert_df(tickers_hist[ticker][item]), mime="text/csv", file_name=ticker+item+".csv")

    tickers_hist[ticker]["year"] = tickers_hist[ticker].index.values

    #     print(tickers_hist[ticker].columns)

    High = tickers_hist[ticker][["year", "High"]]

    Low = tickers_hist[ticker][["year", "Low"]]

    #     print(High.columns,"columns")

    High["type"] = "High"

    print("High.columns:  ", High.columns)

    Low = Low.rename(columns={"Low": "value"})

    Low["type"] = "Low"

    High = High.rename(columns={"High": "value"})

    print("High.columns:  ", High.columns)

    High_Low = pd.concat([High, Low], ignore_index=True)

    #     print("High")

    #     print("Low")

    #     print(Low)

    #     print(High)

    print("High_Low: ", High_Low)

    print("type of high:", type(High))

    fig = px.line(High_Low, x="year", y="value", color="type")

    st.plotly_chart(fig)
    st.download_button('Download all', convert_df(tickers_hist[ticker]), mime="text/csv",
                       file_name=ticker + item + ".csv")
    return High_Low
if screen == 'Open and Close Performance':
    start_date=st.date_input("start_date", datetime_parent.date(2019, 4, 13))
    end_date = st.date_input("end_date", datetime.now())

    if end_date<start_date:
        st.text("End date is before start date!")
    range = round((datetime_parent.date.today() - start_date).days / 365.25 + 0.5)
    if symbol == "":
        st.text("please enter a stock symbol")

    else:
        dataframe_complete = get_dataframe([symbol], period=str(range)+'y',
                                 start_date=start_date,
                                 end_date = end_date
                                 )

        cum_return= give_cum_return(dataframe_complete['Close'])
        st.metric("return", value = "{:.2f}%".format(cum_return*100), delta=None)
        log_std= give_std(dataframe_complete['Close'])
        st.metric('std log',value= "{:.2f}".format(log_std), delta=None)
        dataframe = plot_tickers([symbol], period=str(range)+'y',
                                 start_date=start_date,
                                 end_date = end_date
                                 )
        # data = "value"
        # value_keys = {}
        # for key, name in etf_mapping.items():
        #     value_keys[value] = st.checkbox(name)
        #
        #
        # def compare_with(symbol_comparison):
        #     dataframe_comparison = get_dataframe([symbol_comparison], period=str(range)+'y',
        #                          start_date=start_date,
        #                          end_date = end_date
        #                          )
        #     cum_return_comparison = give_cum_return(dataframe_comparison['Close'])
        #
        #     dataframe_comparison['relative'] = cum_return/cum_return_comparison[0]
        #
        #     px.line(dataframe_comparison, )
        #     dataframe = plot_tickers([symbol], period=str(range) + 'y',
        #                              start_date=start_date,
        #                              end_date=end_date
        #                              )
        #     cum_return_comaprison =

if screen == 'Performance indicators':
    start_date = st.date_input("start_date", datetime_parent.date(2019, 4, 13))
    end_date = st.date_input("end_date", datetime.now())
    start_date = time_to_string(start_date, "%d/%m/%Y")
    end_date = time_to_string(end_date, "%d/%m/%Y")
    if end_date < start_date:
        st.text("End date is before start date!")
    etf = st.radio("Select ETF", ["QQQ", "SPY", "ACWI", "DIA", "IWM", "VPL", "VGK", "KXI", "HEDJ", "USO", "VWO", "XLE"])
    stock = qs_mod.utils.download_returns(symbol)
    qs_mod.stats.sharpe(stock)

    filepath = os.path.join(os.getcwd(), "cache")

    qs_mod_reports.html(stock, etf, start_date =None, end_date=None,
                            output=os.path.join(os.getcwd(), "1.html"),
                            filepath=filepath)

    fig = graphs.create_returns_graph(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.make_cumulative_returns_graph(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.make_log_cumulative_returns_graph(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.cumulative_returns_volatility_graph(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.rolling_sharpe(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.rolling_sortino(filepath, start_date, end_date)
    st.plotly_chart(fig)

    fig = graphs.make_heatmap(filepath, start_date=None, end_date=None)
    st.plotly_chart(fig, use_container_width=True)

    ## that should make use the files in the desired location

    # go to the filepath, load the data, filter the data against our dates, plot the data



    # show sharpe ratio
    #filename = os.path.join('qs_reports', "{}_{}_{}.html".format(symbol, etf, str(int(period_slider))))
    #if not os.path.exists(filename):
        #qs.reports.html(stock, etf, title="{} vs. {}".format(symbol, etf), output=filename)
    #with open(filename, 'r') as file:
        #html = file.read()
    #st.components.v1.html(html,height = 5000, width = 1100, scrolling = False)
    #qs.stats.sharpe(stock)

    # or using extend_pandas() :)

#if screen == 'MA':
    # data = yf.Ticker(symbol).history(period='20y')
    # start_date = st.date_input("start_date", datetime_parent.date(2019, 4, 13))
    # end_date = st.date_input("end_date", datetime.now())
    #
    # ma = 21
    # data['returns'] = data["Close"].diff()
    # data['ma']= data['Close'].rolling(ma).mean()
    # data['ratio']= data['Close'] / data['ma']
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['ratio'], name="stock_open"))
    # fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)

if screen == 'Forecast':
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Forecast App')

    stocks = {symbol}
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.drop(index=0).tail())


    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    #st.download_button('Download', convert_df(tickers_hist[ticker][item]), mime="text/csv",
                      # file_name=ticker + item + ".csv")

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
    forecast.to_csv("/Users/alexfiner/Desktop/forecast/{}.csv".format(selected_stock))
    print("saved")
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.write("*FBProphet is open source software released by Facebook’s Core Data Science team.")
    st.write("The software sets a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.")
    st.write("It works best with time series that have strong seasonal effects and several seasons of historical data")
    st.write("Over the selected range, weekly, yearly, and monthly variance is visualised")


if screen == 'Oil and CPI':
    st.subheader("Oil data")
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    oil = stock.get_oil_prices(st)
    print(oil)
    oildataframe = pd.DataFrame(oil)

    st.subheader("Oil dataframe")
    oildataframe = oildataframe.drop(np.where(oildataframe['value'] == 0)[0])
    oildataframe['days'] = [datetime.utcfromtimestamp(int(str(item)[:-3])) for item in
                                            oildataframe['date']]
    #oildataframe['Date'] = [item[:8] for item in oildataframe['days']]
    oildataframe['Date'] = oildataframe['days'].dt.strftime("%Y-%m-%d")
    fig = px.line(oildataframe, x='days', y="value", hover_data={"days":False, "Date":True})
    st.plotly_chart(fig)

    st.write(oildataframe[["Date", "value", "id", "key"]])

    st.subheader("CPI data")
    st.write("The Consumer Price Index (CPI), the principal gauge of the prices of goods and services")
    st.write("indicates whether the economy is experiencing inflation, deflation or stagflation")
    st.write("Individual investors can also benefit from watching the CPI when making hedging and allocation decisions.")
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    n_years = st.slider('Years', 1, 20)
    cpi = stock.get_cpi_prices(st, n_years)
    print(cpi)
    cpidataframe = pd.DataFrame(cpi)

    cpidataframe = cpidataframe.drop(np.where(cpidataframe['value'] == 0)[0])
    cpidataframe['days'] = [datetime.utcfromtimestamp(int(str(item)[:-3])) for item in
                            cpidataframe['date']]

    cpidataframe['Date'] = cpidataframe['days'].dt.strftime("%Y-%m-%d")

    fig = px.line(cpidataframe, x='days', y="value", hover_data={"days": False, "Date": True})
    st.plotly_chart(fig)

if screen=='FX':
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)

    columns1, columns2, columns3 = st.columns(3)
    with columns1:
        cur_1 = st.selectbox("Currency from", currency_symbol_list)

    with columns2:
        cur_2 = st.selectbox("Currency to", currency_symbol_list)

    with columns3:
        amount = st.number_input("Amount", min_value=0, value=0)
    output = stock.get_fx_currency(cur_1,cur_2,amount)
    output_amount = output[0]['amount']
    rate=output[0]["rate"]
    try:
        if rate==1:
            delta = None
        elif rate<1:
            delta = "-"+str(rate)
        else:
            delta = str(rate)
        delta=None
        with columns1:

            st.metric(label="Amount in {}".format(cur_2), value=output_amount,delta=None)
        with columns2:
            st.metric(label="Exchange rate", value=rate, delta=None)
    except:
        pass

if screen == 'Sector':
    st.text("Daily update on US Sector Performance")
    logo_cache_key = f"{symbol}_logo"
    cached_logo = client.get(logo_cache_key)
    Sector_prices = stock.get_sector_prices(st)

    sector_dataframe = pd.DataFrame(Sector_prices)

    sector_dataframe['Last updated'] = [(datetime.utcfromtimestamp(int(time / 1000)).strftime('%Y-%m-%d %H:%M:%S')) for
                                        time in sector_dataframe['lastUpdated']]
    displayed_x_name = "Sector"
    sector_dataframe[displayed_x_name] = sector_dataframe['name']
    fig = px.bar(sector_dataframe, x=displayed_x_name, y="performance", hover_data=['Last updated'],
                 title="Sector analysis", width=1000, height=800)
    fig.update_layout(font=dict(size=25))#color="Grey"))
    st.plotly_chart(fig)

# if screen == 'Peer Group':
#     peer_group_cache_key = f"{symbol}_peer_group"
#     peer_group = client.get(peer_group_cache_key)
#
#     if peer_group is None:
#         peer_group = stock.get_peer_group()
#         client.set(peer_group_cache_key, json.dumps(peer_group))
#     else:
#         print("getting Peer Group from cache")
#         peer_group = json.loads(peer_group)


















