#====================================================================================================
#                                           Import statements
#====================================================================================================
import pandas as pd
import yfinance as yf

#====================================================================================================
#                                       Retrieve NASDAQ 100 Tickers
#====================================================================================================

# Retrieve datatable
nasdaq100_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
headers = {"User-Agent": "Mozilla/5.0"}
tables = pd.read_html(nasdaq100_url, storage_options=headers)

for table in tables:
    if 'Ticker' in table.columns:
        nasdaq_table = table
        break

# Retrieve tickers
tickers = nasdaq_table['Ticker'].tolist()
tickers = [t.replace('.', '-') for t in tickers]

print("Number of tickers: ", len(tickers))
print(tickers)

#====================================================================================================
#                                  Download historical data for all tickers
#====================================================================================================
start_date = "2021-01-01"
end_date = "2025-12-31"
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True, auto_adjust=False)

# Stores a dataframe for each ticker's historical data.
ticker_historical_dfs = []       

for t in tickers:
    df = data.get(t)

    # Dataframe is null or not found.
    if df is None or df.empty:
        print(f"No data for {t}. Skipping...")
        continue

    # Adder 'Ticker' columns to dataframe.
    df['Ticker'] = t
    df.reset_index(inplace=True)
 
    # Keep only the columns we want from the dataframe, skipping any that are missing. Also reorder columns.
    relevant_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[[c for c in relevant_columns if c in df.columns]]

    ticker_historical_dfs.append(df)

# Combines all tickers' dataframes into one big dataframe.
final_df = pd.concat(ticker_historical_dfs, ignore_index=True)

# Save to a CSV file.
final_df.to_csv('nasdaq100_historicalData.csv', index=False)
print("Historical stock data saved to CSV.")