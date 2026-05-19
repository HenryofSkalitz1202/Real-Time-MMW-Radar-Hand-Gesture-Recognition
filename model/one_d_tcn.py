import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm # Import tqdm for the progress bar

# 1. Depthwise Separable 1D Convolution
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        # Depthwise conv: groups=in_channels applies a single filter per input channel
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, dilation=dilation, 
                                   groups=in_channels, bias=False)
        # Pointwise conv: 1x1 kernel to combine the channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# 2. Causal Padding Utility (Ensures we don't look into the "future")
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 3. A Single Temporal Block
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super().__init__()
        # Padding required for causal convolution: (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        # First DS-Conv Layer
        self.conv1 = DepthwiseSeparableConv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation)
        self.chomp1 = Chomp1d(padding) # Removes padding from the right (future)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second DS-Conv Layer
        self.conv2 = DepthwiseSeparableConv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

        # Residual connection: 1x1 conv if input/output channels differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 4. A Single DS-TCN Branch
class DS_TCN_Branch(nn.Module):
    def __init__(self, in_channels=1, num_channels=16, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        # The paper uses 3 temporal blocks with dilation rates 1, 2, and 4
        dilation_rates = [1, 2, 4]
        
        for i, dilation in enumerate(dilation_rates):
            # Input to the first block is 1 channel (e.g., just Range), subsequent blocks take 16 channels
            input_dim = in_channels if i == 0 else num_channels
            layers.append(TemporalBlock(input_dim, num_channels, kernel_size, stride=1, 
                                        dilation=dilation, dropout=dropout))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 5. Multi-Branch Network (Your 3 Features)
class GestureRecognitionFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        # One independent TCN branch for each feature
        self.range_tcn = DS_TCN_Branch()
        self.doppler_tcn = DS_TCN_Branch()
        self.azimuth_tcn = DS_TCN_Branch()

    def forward(self, range_seq, doppler_seq, azimuth_seq):
        # Pass each 1D sequence through its respective TCN
        out_r = self.range_tcn(range_seq)
        out_d = self.doppler_tcn(doppler_seq)
        out_a = self.azimuth_tcn(azimuth_seq)
        
        # Concatenate outputs along the channel dimension (Stage 2 preparation)
        # Each output is [Batch, 16 channels, 40 frames]
        # Merged output will be [Batch, 48 channels, 40 frames]
        merged_features = torch.cat([out_r, out_d, out_a], dim=1)
        return merged_features

class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 1D Conv across the channel dimension to model interactions
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        y = self.avg_pool(x) # [Batch, Channels, 1]
        
        # PyTorch Conv1d expects [Batch, In_Channels, Spatial_Dim]
        # We swap dimensions so Channels become the "Spatial" dimension for the conv
        y = y.transpose(-1, -2) # [Batch, 1, Channels]
        y = self.conv(y)        # [Batch, 1, Channels]
        y = y.transpose(-1, -2) # [Batch, Channels, 1]
        
        y = self.sigmoid(y)
        # Multiply attention weights with original feature map
        return x * y.expand_as(x)

class DS_CA_Block(nn.Module):
    """Depthwise Separable Channel Attention Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard padding=1 for kernel=3 keeps sequence length intact
        self.ds_conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.eca = ECA(k_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.ds_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.eca(x)
        x = self.maxpool(x)
        return x
    
class GestureRecognitionNetwork(nn.Module):
    """The Complete Two-Stage Architecture """
    def __init__(self, num_classes=12): # You have 12 gestures in your folder!
        super().__init__()
        
        # Stage 1: Feature Extraction
        self.frontend = GestureRecognitionFrontend()
        
        # Stage 2: Channel-wise Fusion (From Table III in the paper)
        self.fusion_block1 = DS_CA_Block(in_channels=48, out_channels=128)
        self.fusion_block2 = DS_CA_Block(in_channels=128, out_channels=128)
        
        self.dropout = nn.Dropout(0.3)
        
        # After two MaxPool1d(2) operations, your 40-frame sequence becomes 10 frames long.
        # Flattened size: 128 channels * 10 frames = 1280
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, range_seq, doppler_seq, azimuth_seq):
        # Stage 1
        x = self.frontend(range_seq, doppler_seq, azimuth_seq) # [Batch, 48, 40]
        
        # Stage 2
        x = self.fusion_block1(x) # [Batch, 128, 20]
        x = self.fusion_block2(x) # [Batch, 128, 10]
        
        # Classification
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten to [Batch, 1280]
        logits = self.fc(x)       # [Batch, 12]
        
        return logits