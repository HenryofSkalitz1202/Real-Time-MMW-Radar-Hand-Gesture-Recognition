import torch
import torch.nn as nn

class LSTM_Gesture_Network(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, num_classes=12):
        super().__init__()
        
        # Based on the paper: 2 hidden layers with 32 neurons each.
        # batch_first=True ensures input shape is [batch, seq_len, features]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 # Standard dropout between LSTM layers to prevent overfitting
        )
        
        # The final fully connected layer maps the hidden state to the gesture classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, range_seq, vel_seq, az_seq):
        # 1. Combine your 3 feature branches into a single tensor
        # Shape: [batch, 3, 40]
        x = torch.cat([range_seq, vel_seq, az_seq], dim=1)
        
        # 2. Transpose to match LSTM requirements
        # Shape becomes: [batch, 40, 3] (Batch, Sequence_Length, Features)
        x = x.transpose(1, 2)
        
        # 3. Pass through the LSTM
        # lstm_out contains all hidden states across the sequence
        # h_n contains the final hidden state for each layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 4. Extract the hidden state from the very last time step of the last layer
        # h_n shape: [num_layers, batch, hidden_size]
        last_hidden = h_n[-1, :, :] 
        
        # 5. Classify the gesture
        logits = self.fc(last_hidden)
        
        return logits