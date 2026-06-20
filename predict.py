#====================================================================================================
#                                           Import statements
#====================================================================================================
import yfinance as yf
import numpy as np
import torch
import pickle

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models import LSTM

#====================================================================================================
#                                               Config
#====================================================================================================
MODEL_PATH = 'LSTM_Model.pth'
SEQUENCE_LEN = 60
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10Days', "Volatility_10Days", 'Sentiment']

#====================================================================================================
#                                         User Input: Ticker Symbol
#====================================================================================================
ticker = input("Enter ticker symbol: ").upper().strip()
print(f"Prediciting tomorrow's stock price for {ticker}...")

#====================================================================================================
#                                      Download Recent Stock Price Data
#====================================================================================================
# ===== Download the data =====
ticker_df = yf.download(ticker, period='120d', auto_adjust=False)               # Have >60 days to accomadate for SMA and Volatility features, 
                                                                                # and to have an extra buffer to account for NaN, weekends, etc.
# ===== Empty dataframe check =====
if ticker_df.empty:                                                             
    print(f"No price data found for ticker symbol: {ticker}")
    exit()

# ===== Cleaning up the datframe =====
ticker_df = ticker_df.reset_index()

ticker_df.columns = [column[0] if isinstance(column, tuple) else column for column in ticker_df.columns]

ticker_df.columns.name = None

ticker_df = ticker_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

#====================================================================================================
#                                      Calculate Features
#====================================================================================================
# ===== SMA =====
ticker_df['SMA_10Days'] = ticker_df['Close'].rolling(window=10).mean()

# ===== Volatility =====
ticker_df['Return'] = ticker_df['Close'].pct_change()
ticker_df['Volatility_10Days'] = ticker_df['Return'].rolling(window=10).std()

# ===== Sentiment =====
ticker_obj = yf.Ticker(ticker)
ticker_news = ticker_obj.news

vader_analyzer = SentimentIntensityAnalyzer()

if ticker_news:
    article_scores = []
    for article in ticker_news:
        content = article.get('content', {})
        article_title = str(content.get('title', ''))
        score = vader_analyzer.polarity_scores(article_title)['compound']
        article_scores.append(score)
    
    sentiment = sum(article_scores)/len(article_scores)                                 
    print(f"Sentiment score from {len(article_scores)} articles: {sentiment:.3f}")
else:
    sentiment = 0.0
    print(f"No recent news found for ticker {ticker}. Using netural (0.0) sentiment!")

ticker_df['Sentiment'] = sentiment                                                                      # Sentiment values for all days will be the same.

# ===== Drop rows that contian NaN =====
ticker_df = ticker_df.dropna().reset_index(drop=True)

#====================================================================================================
#                                      Scale Data
#====================================================================================================
with open('scaler.pkl', 'rb') as f:                 # Get the scaler used before.
    scaler = pickle.load(f)

scaled = scaler.transform(ticker_df[FEATURES])      # Scale data.

#====================================================================================================
#                                    Create the Sequence
#====================================================================================================
# ===== Amount of data check =====
if len(scaled) < SEQUENCE_LEN:
    print("Not enough days worth of data!")
    print(f"Need at least {SEQUENCE_LEN} days worth of data.")
    print(f"Currently had {len(scaled)} days worth of data.")
    exit()
    
# ===== Get the last 60 days worth of data only =====
sequence = scaled[-SEQUENCE_LEN:]
sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

#====================================================================================================
#                                    Load LSTM Model and Predict
#====================================================================================================
input_size = len(FEATURES)
model = LSTM(input_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    prediction_scaled = model(sequence_tensor)

#====================================================================================================
#                                     Obtain the Price Prediction 
#====================================================================================================
# The final price prediction is the stock price the next day.
# Previous section produced a price prediction as a scaled value. Get the dollar value now.

dummyArray = np.zeros((1, len(FEATURES)))
dummyArray[0,3] = prediction_scaled.item()                          # Index 3 is the Close Price.
prediction_price = scaler.inverse_transform(dummyArray)[0,3]

#====================================================================================================
#                                     Output the Price Prediction 
#====================================================================================================
last_closing_price = ticker_df['Close'].iloc[-1]

print(f"\nTicker: {ticker}")
print(f"Last closing price: ${last_closing_price:.2f}")
print(f"Tomorrow's predicted price: ${prediction_price:.2f}")
