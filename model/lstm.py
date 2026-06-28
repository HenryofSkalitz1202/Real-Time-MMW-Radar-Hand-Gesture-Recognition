import torch
import torch.nn as nn
import torch.nn.functional as F

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) module inspired by Kim et al.
    Applies a local 1D convolution across the channels to learn their interdependencies
    without dimensionality reduction.
    """
    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        # Global Average Pooling across the feature dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 1D Conv across the 4 channels (R, V, A, E)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, 4_channels, hidden_size]
        y = self.avg_pool(x) # Result: [batch, 4, 1]
        
        # Transpose to [batch, 1, 4] so Conv1D operates across the 4 channels
        y = y.transpose(-1, -2) 
        y = self.conv(y)        # Result: [batch, 1, 4]
        y = y.transpose(-1, -2) # Transpose back to [batch, 4, 1]
        
        # Multiply the dynamic attention weights back into the original features
        return x * self.sigmoid(y)

class LSTM_Gesture_Network(nn.Module):
    def __init__(self, branch_hidden_size=16, num_layers=1, num_classes=6):
        super().__init__()
        
        # 1. CHANNELWISE SEPARATION
        # Instead of one LSTM with input_size=4, we use 4 parallel LSTMs with input_size=1.
        # branch_hidden_size is set to 16 to keep total parameters lightweight.
        self.lstm_r = nn.LSTM(input_size=1, hidden_size=branch_hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_v = nn.LSTM(input_size=1, hidden_size=branch_hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_a = nn.LSTM(input_size=1, hidden_size=branch_hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_e = nn.LSTM(input_size=1, hidden_size=branch_hidden_size, num_layers=num_layers, batch_first=True)
        
        # 2. CHANNELWISE FUSION
        # ECA layer to route attention between Range, Velocity, Azimuth, and Elevation
        self.eca = ECALayer(k_size=3) 
        
        # 3. CLASSIFICATION
        # The flattened output of 4 branches * 16 hidden neurons = 64
        self.fc = nn.Linear(4 * branch_hidden_size, num_classes)

    def forward(self, range_seq, vel_seq, az_seq, el_seq):
        # Transpose inputs from [batch, 1, 40] to [batch, 40, 1] for LSTMs
        r = range_seq.transpose(1, 2)
        v = vel_seq.transpose(1, 2)
        a = az_seq.transpose(1, 2)
        e = el_seq.transpose(1, 2)
        
        # Pass each kinematic parameter through its own dedicated LSTM branch
        _, (h_r, _) = self.lstm_r(r)
        _, (h_v, _) = self.lstm_v(v)
        _, (h_a, _) = self.lstm_a(a)
        _, (h_e, _) = self.lstm_e(e)
        
        # Extract the final hidden state from each branch's last layer
        # Resulting shape per branch: [batch, branch_hidden_size]
        h_r = h_r[-1]
        h_v = h_v[-1]
        h_a = h_a[-1]
        h_e = h_e[-1]
        
        # Stack them to form a channel dimension for the attention layer
        # Shape becomes: [batch_size, 4, branch_hidden_size]
        stacked_features = torch.stack([h_r, h_v, h_a, h_e], dim=1)
        
        # Apply Channel Attention
        # If a gesture is purely lateral (Swipe Left/Right), the ECA will automatically 
        # down-weight the Elevation channel's noise for this specific inference.
        attended_features = self.eca(stacked_features)
        
        # Flatten the attended features to [batch_size, 64]
        flattened = attended_features.view(attended_features.size(0), -1)
        
        # Final fully connected classification
        logits = self.fc(flattened)
        
        return logits