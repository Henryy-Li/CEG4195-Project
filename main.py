#====================================================================================================
#                                           Import Statements
#====================================================================================================
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models import LSTM, GRU, Transformer
from dataset import sequence, split_data, scale_data, create_loader

#====================================================================================================
#                                       Retrieve NASDAQ 100 Tickers
#====================================================================================================
# ===== Retrieve all tables on the Wikipedia article =====
nasdaq100_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
headers = {"User-Agent": "Mozilla/5.0"}
tables = pd.read_html(nasdaq100_url, storage_options=headers)
print("Completed: Tables retrieved!")

# ===== Find the table that list all tickers =====
for table in tables:
    if 'Ticker' in table.columns:
        nasdaq_table = table
        break
print("Completed: Ticker table found!")

# ===== Retrieve tickers list =====
tickers = nasdaq_table['Ticker'].tolist()
tickers = [t.replace('.', '-') for t in tickers]                # Replace "." for "-" in ticker names.
print("Completed: Ticker list retrieved!")

#====================================================================================================
#                                  Download Historical Data for All Tickers
#====================================================================================================
# ===== All of the tickers' data =====
ticker_df_list = []     

# ===== Download the data for the time period =====
start_date = "2010-01-01"
end_date = "2019-12-31"
data = yf.download(tickers, start = start_date, end = end_date, group_by = 'ticker', threads = True, auto_adjust = False)
print("Completed: Data downloaded!")

# ===== Process data for each ticker =====
for ticker in tickers:
    try:
        ticker_df = data[ticker].copy()                                                   # Copy the data for this ticker.
    except KeyError:                            
        print(f"No data for {ticker}. Skipped!")                                          # Dataframe not found.
        continue
    if ticker_df.empty:                                
        print(f"No data for {ticker}. Skipped!")                                          # Dataframe is null.
        continue

    ticker_df['Ticker'] = ticker                                                          # Added 'Ticker' column to dataframe.
    ticker_df = ticker_df.reset_index()                                                   # Remove the date from being the index.
    ticker_df = ticker_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]   # Reorder columns
    ticker_df_list.append(ticker_df)                                                      # Add ticker dataframe to the list that stores them.

# ===== final_df =====
# Stores all tickers and all feature data.
# - Rows: Every trading day for every ticker.
# - Columns: Features for every row.
# - Specifically here: Combines the list of tickers dataframes into one big dataframe.
final_df = pd.concat(ticker_df_list, ignore_index = True)                       

# ===== Save data to a CSV file =====
final_df.to_csv('NASDAQ_100_Data.csv', index = False)
print("Completed: Data saved to a CSV!")

#====================================================================================================
#                                  Dataset Cleaning & Preprocessing 
#====================================================================================================
final_df['Date'] = pd.to_datetime(final_df['Date'])                             # Change string dates to datetime objects
final_df = final_df.sort_values(['Ticker', 'Date'])                             # Sort by ticker and then by date
final_df = final_df.ffill()                                                     # Forward fill to cover missing values. No prices on weekends & holidays.

final_df = final_df.reset_index(drop=True)                                      # Reset index to 0, 1, ..., N and discard the index.
final_df.columns.name = None                                                    # Remove leftover all encompassing column label.

print("Completed: Data prepocessed!")

#====================================================================================================
#                              Feature Engineering - Create the Features
#====================================================================================================
# ===== Closing price series =====
closing_price_series = final_df.groupby('Ticker')['Close']

# ===== Target Price =====
# The actual next day price. Usage is to use today's closing price to predict tomorrow's closing price.
final_df['Target'] = closing_price_series.shift(-1)

# ===== Simple Moving Average (SMA) =====
# SMA for the last 10 days:
# Average closing price over the last N days.
# Take the last N closing prices, sum up, divide by N.
# The next day, you drop the oldest price, add the newest one, then recalculate.
final_df['SMA_10Days'] = closing_price_series.transform(
    lambda x: x.rolling(window=10).mean()
)

# ===== Volatility =====
# Volatility for the last 10 days:
# The amount the stock price is moving around. 
# A measure of risk and uncertainty.
final_df['Return'] = closing_price_series.transform('pct_change')
final_df['Volatility_10Days'] = final_df.groupby('Ticker')['Return'].transform(
    lambda x: x.rolling(window=10).std()
)

'''
# ===== Relative Strength Index (RSI) =====
# Indicator if a stock is overbought or oversold.
# closing_prices: Closing prices of a ticker.
# window=14 days: Standard RSI period.
def calculate_RSI(closing_prices, window=14):
    daily_price_change = closing_prices.diff()                        
    
    gain = daily_price_change.where(daily_price_change>0,0)
    loss = -daily_price_change.where(daily_price_change<0,0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    relative_strength = avg_gain/avg_loss
    relative_strength_index = 100 - (100/(1+relative_strength))
    return relative_strength_index

final_df['RSI'] = closing_price_series.transform(calculate_RSI)

# ========== Moving Average Convergence Divergence (MACD) ==========
# Momentum of a stock. 
# Expoential Moving Average (EMA):
# - Like SMA but places heavier weights on more recent items.
# - Fast EMAs: Shorter memory. Recent price dominante.
# - Slow EMAs: Longer memory. Older prices have more influence.
# - 12 26 9 is the universal standard for MACD.

def calculate_MACD(closing_prices, fast=12, slow=26, signal=9):
    EMA_fast = closing_prices.ewm(span=fast).mean()                       
    EMA_slow = closing_prices.ewm(span=slow).mean()
    MACD_line = EMA_fast - EMA_slow                                 # Shows price changes. Indicates rising (bullish) or falling (bearish) prices.
    signal_line = MACD_line.ewm(span=signal).mean()                 # Smoothes out the MACD line.
    histogram_line = MACD_line - signal_line                        # Shows momentum acceleration. Shows increasing (bullish) or decreasing (bearish) momentum.
    MACD_dataFrame = pd.DataFrame({                                 # MACD => Velocity, Histogram => Accleration.
        'MACD_Line':      MACD_line,
        'MACD_Signal':    signal_line,
        'MACD_Histogram': histogram_line
    }) 
    return MACD_dataFrame 

MACD_df = closing_price_series.apply(calculate_MACD).reset_index(level=0, drop=True)
final_df['MACD_Line'] = MACD_df['MACD_Line']
final_df['MACD_Signal'] = MACD_df['MACD_Signal']
final_df['MACD_Histogram'] = MACD_df['MACD_Histogram']
'''

#====================================================================================================
#                    Feature Engineering Continued - NLP For Stock News Headlines
#====================================================================================================
# ===== Loading Dataset =====
news_df = pd.read_csv('analyst_ratings_processed.csv', index_col=0)    
print("Completed: News headlines dataset loaded!")

news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True)                # Convert date column values to datetime objects.  
news_df = news_df.dropna(subset=['date'])                                                   # Non-datetime values are converted to NaT (not a time) and are dropped.

# ===== Calculating Sentiment on Each Headline =====
vader_analyzer = SentimentIntensityAnalyzer()                                    
news_df['Sentiment'] = news_df['title'].apply(
    lambda x: vader_analyzer.polarity_scores(str(x))['compound']
)
daily_sentiment = news_df.groupby(['stock', 'date'])['Sentiment'].mean().reset_index()      # Calculate the average sentiment per day, per ticker.
print("Completed: Sentiment calculated on news headlines!")

daily_sentiment = daily_sentiment.rename(columns={'stock': 'Ticker', 'date': 'Date'})                       
daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'].dt.date)                   # Ensure Date is still a datetime object after groupby. 
                                                                                            # Take only the date and set time to 00:00.        
# ===== Merge Sentiments with Final Dataframe =====
final_df = final_df.merge(
    daily_sentiment,
    on = ['Ticker', 'Date'],
    how = 'left'
)
final_df['Sentiment'] = final_df['Sentiment'].fillna(0)
print("Completed: Merged final dataframe with sentiment dataframe!")

# ========== Drop NaN Rows ==========
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10Days', 'Volatility_10Days', 'Sentiment']
final_df = final_df.dropna(subset=features)
final_df = final_df.reset_index(drop=True)

print("Completed: Feature engineering done!")

#====================================================================================================
#                                    Data Preparation
#====================================================================================================
# ===== Train-Test Splits =====
train_df, test_df = split_data(final_df)

# ===== Data Scaling =====
train_scaled, test_scaled, scaler = scale_data(train_df, test_df, features)

with open('scaler.pkl', 'wb') as f:                                                 # Save scaler.
    pickle.dump(scaler, f)
print("Completed: Scaler saved!")

# ===== Sequences =====
sequence_len = 60                                                                   # 60 trading days (~ 3 months)
X_train, y_train = sequence(train_scaled, sequence_len) 
X_test, y_test = sequence(test_scaled, sequence_len) 

# ===== Data Loader =====
train_loader, X_test_tensor, y_test_tensor = create_loader(X_train, y_train, X_test, y_test)

print("Completed: Data preparation done!")

#====================================================================================================
#                                    LSTM Training
#====================================================================================================

print("LSTM training starting...")

# ===== Intializing LSTM Model =====
input_size = X_train.shape[2]                                           # Gets number of features (8)
model = LSTM(input_size)
mse = nn.MSELoss()                                                      # Mean Sqaured Error (MSE) loss object. Calculates MSE when called.

# ====== Train or Load a LSTM Model =====
if os.path.exists('LSTM_Model.pth'):                                    # Load in previously trained model.
    model.load_state_dict(torch.load('LSTM_Model.pth'))
    model.eval()
    print("Saved model found and loaded! Training skipped!")    
else:                                                                   # Train model for the first time.
    # ====== MSE and Optimizer ======
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)         # Adjusts weights during training.

    # ====== Training Loop ======
    epochs = 20
    for epoch in range(epochs):
        model.train()                       # Model set to training mode.                                            
        epoch_loss = 0                      # Store total loss for current epoch.

        for xb, yb in train_loader:         # Loop through all training batches (b)
            optimizer.zero_grad()           # Set gradients from previous batch to 0.
            
            predictions = model(xb)         # Prediction

            loss = mse(predictions,yb)      # MSE
            loss.backward()                 # Backpropagation to compute gradients for every weight.
            optimizer.step()                # Use those gradients to update model weights to reduce loss.
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} of {epochs}. Avg loss per batch: {epoch_loss/len(train_loader):.6f}")
    
    # ===== Save Model ===== (Avoid retraining model every time program is run)
    torch.save(model.state_dict(), 'LSTM_Model.pth')          # Saves the weights of the model.
    print("Completed: LSTM model saved!")

print("Completed: LSTM trained!")

#====================================================================================================
#                                    LSTM Evaluation
#====================================================================================================

print("LSTM testing starting...")

# ===== Testing model on unseen data =====
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    loss = mse(predictions, y_test_tensor)
    print(f"Test loss: {loss.item():.6f}")

print("Completed: LSTM tested!")

# ===== Convert scaler values to real prices =====
# Inverse transform expects 8 features but predictions and y_test are only 1 feature, closing price.
# Create a dummy array (DA) to allow for the inverse transformation.
predictions_DA = np.zeros((len(predictions), len(features)))
predictions_DA[:,3] = predictions.flatten()                                     # Place closing prices in the corect column.
predictions_prices = scaler.inverse_transform(predictions_DA)[:,3]              # Convert and select only the closing price column.

y_test_DA = np.zeros((len(y_test), len(features)))               
y_test_DA[:,3] = y_test                                        
y_test_prices = scaler.inverse_transform(y_test_DA)[:,3]       

# ===== Metrics =====
MAE_prices = mean_absolute_error(y_test_prices, predictions_prices)             # Mean Absolute Error (MAE)
MSE_prices = mean_squared_error(y_test_prices, predictions_prices)              # Mean Squared Error (MSE)
RMSE_prices = math.sqrt(MSE_prices)                                             # Root Mean Squared Error (RMSE)

print(f"Mean Absolute Error: ${MAE_prices:.2f}")
print(f"Mean Squared Error: ${MSE_prices:.2f}")
print(f"Root Mean Squared Error: ${RMSE_prices:.2f}")

# ===== Graph predicted prices vs actual prices =====
plt.title('Predicted vs Actual Stock Prices')
plt.xlabel('Index of the Test Sample')
plt.ylabel('Stock Price ($)')

plt.plot(predictions_prices, label='Predicted Prices')
plt.plot(y_test_prices, label='Actual Prices')

plt.legend()
plt.show()
