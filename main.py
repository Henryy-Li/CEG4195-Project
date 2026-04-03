#====================================================================================================
#                                           Import statements
#====================================================================================================
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

#====================================================================================================
#                                              Functions 
#====================================================================================================
def simple_moving_average(group, window=10):
    group['SMA_10Days'] = group['Close'].rolling(window=window).mean()
    return group

def volatility(group, window=10):
    group['Return'] = group['Close'].pct_change()
    group['Volatility_10Days'] = group['Return'].rolling(window=window).std()
    return group

def sequence(data, sequence_len):
    X = []
    y = []

    for i in range(len(data) - sequence_len):
        X.append(data[i:i+sequence_len])    
        y.append(data[i+sequence_len, 3])           # 3 represents the 3rd indexed column, price.
    return np.array(X), np.array(y)

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

#====================================================================================================
#                                  Download historical data for all tickers
#====================================================================================================
start_date = "2021-01-01"
end_date = "2025-12-31"
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True, auto_adjust=False)

# Stores a dataframe for each ticker's historical data.
ticker_historical_dfs = []       

for t in tickers:
    try:
        df = data[t].copy()
    except KeyError:                            # Dataframe not found.
        print(f"No data for {t}. Skipped!")
        continue
    if df.empty:                                # Dataframe is null.
        print(f"No data for {t}. Skipped!")
        continue

    df['Ticker'] = t                            # Added to 'Ticker' column in dataframe.
    df = df.reset_index()

    # Keep only the columns we want from the dataframe, skipping any that are missing. Also reorder columns.
    relevant_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[[c for c in relevant_columns if c in df.columns]]

    ticker_historical_dfs.append(df)

# Combines all tickers' dataframes into one big dataframe.
final_df = pd.concat(ticker_historical_dfs, ignore_index=True)

# Save to a CSV file.
final_df.to_csv('nasdaq100_historicalData.csv', index=False)
print("Historical stock data saved to CSV.")

#====================================================================================================
#                                  Dataset Preprocessing 
#====================================================================================================
final_df['Date'] = pd.to_datetime(final_df['Date'])                             # Change string dates to datetime objects
final_df = final_df.sort_values(['Ticker', 'Date'])                             # Sort by ticker and then by date
final_df = final_df.groupby('Ticker', as_index=False).ffill()                   # Forward fill

#====================================================================================================
#                                    Feature Engineering - Create the Features
#====================================================================================================
# Feature: Daily returns
# Method: Use today's closing price to predict tomorrow's closing price.
final_df['Target'] = final_df.groupby('Ticker')['Close'].shift(-1)

# Feature: Simple moving average for last 10 days  
final_df = final_df.groupby('Ticker').apply(simple_moving_average).reset_index(drop=True)

# Feature: Volatility for the last 10 days
final_df = final_df.groupby('Ticker').apply(volatility).reset_index(drop=True)

#====================================================================================================
#                                    Train-Test Splits
#====================================================================================================
# Time period is 5 years. 
# Halfway point is 2.5 years from the start (2021-01-01)

train_df = final_df[final_df['Date'] < '2023-07-01']
test_df = final_df[final_df['Date'] >= '2023-07-01']

#====================================================================================================
#                                    Data Scaling
#====================================================================================================
scaler = MinMaxScaler()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

#====================================================================================================
#                                    Sequences
#====================================================================================================
sequence_len = 20        # 20 trading days in 1 month.
X_train, y_train = sequence(train_scaled, sequence_len)
X_test, y_test = sequence(test_scaled, sequence_len)

#====================================================================================================
#                                    LSTM Training
#====================================================================================================

# ====== Create the tensors, dataset, and loader =====
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ====== LSTM Model =====
class LSTM(nn.Module):          
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTM, self).__init__()                                                        # Call the parent class's constructor.

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)          # LSTM layer.
        self.fc = nn.Linear(hidden_size, 1)                                                 # Fully connected layer. It outputs the price prediction.

    def forward(self, inputTensorSeq):
        output, _ = self.lstm(inputTensorSeq)           
        output = output[:,-1,:]                                                             # Select only the last timestamp of each sequence. Keep all features (high, low, etc.)
        output = self.fc(output)                                                            # Prediction
        return output
    
# ====== Intializing LSTM Model =====
input_size = X_train.shape[2]
model = LSTM(input_size)
mse = nn.MSELoss()                                                                          # Used to iteratively improve model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                                  # Used to iteratively improve model

epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()           # Set old gradients to 0.
        predictions = model(xb)

        loss = mse(predictions,yb)      # Mean squared error
        loss.backward()                 # Backpropagation to compute gradients.
        optimizer.step()                # Adjusts model parameters (weights) to reduce loss.
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} of {epochs}. Loss: {epoch_loss/len(train_loader):.3f}")