#==================================================
#                  Import statements
#==================================================
#import yfiance as yf 
import pandas as pd

from pytickersymbols import PyTickerSymbols


#==================================================
#                Load Dataset
#==================================================

# Company stock codes of the top 100 companies on the NASDAQ
tickers_obj = PyTickerSymbols()
nasdaq_100 = tickers_obj._get_tickers_by_index(
    "NASDAQ 100",
    ["NASDAQ"],
    "STOCK"
    )
tickers = []
for x in nasdaq_100:
    tickers.append(x['symbol'])
print(len(tickers), tickers)