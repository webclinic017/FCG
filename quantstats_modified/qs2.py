import os 

import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns('FB')

# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
stock.sharpe()

qs.reports.html(stock, "SPY",  output=os.path.join(os.getcwd(),'quantstats-tearsheet.html'))