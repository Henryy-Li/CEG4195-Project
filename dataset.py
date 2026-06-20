import numpy as np
import torch 

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

#====================================================================================================
#                                           sequence
#====================================================================================================
# X: Get the sequences of data used to predict the next day's closing price.
# y: Get the next day's closing price.

def sequence(data, sequence_len):
    X = []
    y = []
 
    for i in range(len(data) - sequence_len):
        X.append(data[i:i+sequence_len])            # Grab a sequence that is 'sequence_len' rows of data.
        y.append(data[i+sequence_len, 3])           # Grab the closing price that is the next row after the sequence. (3rd indexed column is the closing price)
    return np.array(X), np.array(y)

#====================================================================================================
#                                           split_data
#==================================================================================================== 
# Split data into the train and test sets.
# Time period: 10 years.
# Start: 2010-01-01
# End: 2019-12-31

def split_data(final_df):
    train_df = final_df[final_df['Date'] < '2016-01-01']
    test_df = final_df[final_df['Date'] >= '2016-01-01']
    return train_df, test_df

#====================================================================================================
#                                           scale_data
#==================================================================================================== 
# Scale all the train and test data to a range of 0 to 1.                                  

def scale_data(train_df, test_df, features):
    scaler = MinMaxScaler()                                             # Scaler object.
    train_scaled = scaler.fit_transform(train_df[features])             # Apply scaling
    test_scaled = scaler.transform(test_df[features])                   # Apply scaling using same max/min values learned in the training data.
    return train_scaled, test_scaled, scaler

#====================================================================================================
#                                         create_loader               
#==================================================================================================== 
# Convert numpy arrays to tensors.
# Create the loader that outputs the dataset in batches.   
                           
def create_loader(X_train, y_train, X_test, y_test, batch_size=64):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)                # unsqueeze(1): Change shape from (samples) to (samples, 1) to be able to calculate loss between predictions and targets.
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)                               # Predictions are of the form (samples, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)                           # Match X tensors with their y tensors. Allows for shuffling.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)           # Generates batches of the dataset.
    return train_loader, X_test_tensor, y_test_tensor