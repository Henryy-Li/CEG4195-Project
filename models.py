#====================================================================================================
#                                       Import Statements
#====================================================================================================
import torch
import torch.nn as nn
import math

#====================================================================================================
#                                       LSTM Model
#====================================================================================================
class LSTM(nn.Module):          
    # input_size: Features per timestamp.
    # hidden_size: # memory units of the LSTM.
    # num_layers: # of LSTM layers stacked 
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()                                                                  # Call the parent class's constructor.

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)          # Initalize LSTM both layers.
        self.fc = nn.Linear(hidden_size, 1)                                                 # Fully connected layer (FC). Outputs the price prediction (tomorrow's price).

    def forward(self, inputBatch):
        output, _ = self.lstm(inputBatch)           
        output = output[:,-1,:]                                                             # Select only the last day of each sequence while keeping all features.
        output = self.fc(output)                                                            # Convert hidden state --> Price prediction
        return output

#====================================================================================================
#                                       GRU Model
#====================================================================================================
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputBatch):
        output, _ = self.gru(inputBatch)           
        output = output[:,-1,:]                                                            
        output = self.fc(output)                                                           
        return output
    
#====================================================================================================
#                  Poisitional Encoding Class (support for Transformer Model)
#====================================================================================================
# Transformers process all sequential data at once. 
# It has no inherent sense of order.
# Add position information to each timestep so the transformer knows the order of the days.

class PositionalEncoding(nn.Module):
    # d_model: 
    # - Size of the internal representation of each day's data. 
    # - Number of dimensions. Will change from the 8 features/dimensions to 64.

    # max_len:
    # - Max sequence length. 
    # - Max number of days for the sequence in positional encoding

    # dropout: 
    # - Randomly zero out a percentage of values to prevent overfitting.
    # - The 64 dimensions in one day of data each have a 10% chance of being dropped.
    # - Which is around 10% of the dimensions in a day being dropped.

    # ========== Constructor ==========
    # Define the positional encoding 
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)                                                            # Dropout layer.

        matrix = torch.zeros(max_len, d_model)                                                          # Positional encoding matrix
        
        tensor_1D = torch.arange(0, max_len)                                                            # 1D tensor of numbers from 0 to max_len. Shape: (500,).
        tensor_position = tensor_1D.unsqueeze(1).float()                                                # Convert to a 2D tensor. Shape: (500, 1).
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))   # Frequency scaling factors. Each pair of dimensions gets a different frequency.

        matrix[:, 0::2] = torch.sin(tensor_position*division_term)                                      # Unique wave value for each position-dimension combination.
        matrix[:, 1::2] = torch.cos(tensor_position*division_term)                                      # For all rows, even columns have sin values, odd columns have cos values.
        matrix = matrix.unsqueeze(0)                                                                    # Add the batch dimension to allow for broadcasts.          

        self.register_buffer('matrix', matrix)                                                          # Add matrix to register buffers to be saved and then loaded with the model.

    # ========== Forward Function ==========
    # Add positional encoding (dates) to input.
    # x is a temporary varaible.
    def forward(self, inputBatch_projected):
        x = inputBatch_projected + self.matrix[:, :inputBatch_projected.size(1), :]                     # Add positional encoding to input.
        x = self.dropout(x)
        return x

#====================================================================================================
#                                       Transformer
#====================================================================================================
class Transformer(nn.Module):
    # ========== Constructor ==========
    # Define the transformer model.
    # nhead: Number of attention heads. The number of searches for patterns occurring in parallel.
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)                                          # Projection of the 8 input features to 64 dimensions. Transformer expects an input of 64 dimensions.
        
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)                         # Positional encoding.
        
        encoder_layer = nn.TransformerEncoderLayer(                                                     # Encoding layer.
            d_model = d_model,  
            nhead = nhead,
            dim_feedforward = d_model*4,                                                                # Feed forward network.
            dropout = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)                  # Replicate the layers.

        self.fc = nn.Linear(d_model, 1)                                                                 # Fully connected output layer. 1 output, price.

    # ========== Forward Function ==========
    # The pipeline that data flows through for the transformer model.
    # x is a temporary variable.
    def forward(self, inputBatch):
        x = self.input_projection(inputBatch)           # Projection from 8 to 64 dimensions.
        x = self.positional_encoding(x)                 # Add positional encoding
        x = self.transformer(x)                         # Pass data through the stacked encoder layers. Learn patterns.
        x = x[:, -1, :]                                 # Retrieve the last timestamps, day 60s.
        x = self.fc(x)                                  # Compress 64 dimensions to 1 price prediction.
        return x