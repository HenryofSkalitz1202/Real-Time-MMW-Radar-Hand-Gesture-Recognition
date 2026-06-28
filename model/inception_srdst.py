import torch
import torch.nn as nn
import math

class DSConv1d(nn.Module):
    """Depthwise Separable 1D Convolution for lightweight local feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # Calculate padding to maintain sequence length: (k-1) * d / 2
        padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise step: filters each input channel separately
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, 
            groups=in_channels, bias=False
        )
        # Pointwise step: combines the outputs of the depthwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class TCNBranch(nn.Module):
    """A 3-layer TCN branch dedicated to a single physical feature (e.g., Range)."""
    def __init__(self, out_channels=16):
        super().__init__()
        # Layer 1: Standard conv since input is just 1 channel
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        # Layer 2 & 3: DS Convolutions to expand the receptive field efficiently
        self.layer2 = DSConv1d(out_channels, out_channels, kernel_size=3, dilation=2)
        self.layer3 = DSConv1d(out_channels, out_channels, kernel_size=3, dilation=4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ECABlock(nn.Module):
    """Efficient Channel Attention to dynamically weigh feature importance."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Dynamically calculate 1D kernel size based on channel count
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, T)
        y = self.avg_pool(x) # Pools temporal dimension: (B, C, 1)
        # ECA applies 1D conv across the channel dimension
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2) 
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class PositionalEncoding(nn.Module):
    """Standard Sine/Cosine positional encoding for the Transformer."""
    def __init__(self, d_model, max_len=40):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (B, T, C)
        return x + self.pe[:, :x.size(1), :]


class InceptionSRDST(nn.Module):
    """
    Local-Global 4-Branch Transformer for real-time mmWave Hand Gesture Recognition.
    Target Edge Unit: Raspberry Pi 4b (Optimized for low FLOPs/Params).
    """
    def __init__(self, num_classes=6, branch_out=16, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. Local Temporal Encoders (The 'Inception' / TCN Phase)
        self.branch_range = TCNBranch(branch_out)
        self.branch_doppler = TCNBranch(branch_out)
        self.branch_azimuth = TCNBranch(branch_out)
        self.branch_elevation = TCNBranch(branch_out)
        
        # 2. Tokenization & Embedding
        # 4 branches * 16 channels = 64 local features
        self.projection = nn.Conv1d(4 * branch_out, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_model, max_len=40)
        
        # 3. Global Transformer Encoder (The SRDST Phase)
        # Using a shallow transformer (num_layers=2) to keep inference lightning fast
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, # Kept compact, typically 4x d_model
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Channel-Aware Fusion (ECA)
        self.eca = ECABlock(d_model)
        
        # 5. Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Paper Params
        # self.dropout = nn.Dropout(0.3)

        # Gem Params
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, range_seq, vel_seq, az_seq, el_seq):
        # 1. Dimension Safety Check
        # If the dataloader yields [Batch, 40], we add the channel dimension to make it [Batch, 1, 40]
        if range_seq.dim() == 2:
            range_seq = range_seq.unsqueeze(1)
            vel_seq = vel_seq.unsqueeze(1)
            az_seq = az_seq.unsqueeze(1)
            el_seq = el_seq.unsqueeze(1)
            
        # 2. Isolate the 4 feature rows and pass through independent local branches
        x_r = self.branch_range(range_seq)
        x_d = self.branch_doppler(vel_seq)
        x_a = self.branch_azimuth(az_seq)
        x_e = self.branch_elevation(el_seq)
        
        # 3. Concatenate into dense local feature map: (B, 64, 40)
        x_local = torch.cat([x_r, x_d, x_a, x_e], dim=1)
        
        # 4. Project to Transformer dimension: (B, 128, 40)
        x_proj = self.projection(x_local)
        
        # 5. Sequence formatting for Transformer: (B, 40, 128)
        x_seq = x_proj.transpose(1, 2)
        x_seq = self.pos_encoder(x_seq)
        
        # 6. Apply Multi-Head Self-Attention over the 40 time frames
        x_trans = self.transformer(x_seq)
        
        # 7. Revert sequence formatting for ECA and pooling: (B, 128, 40)
        x_trans = x_trans.transpose(1, 2)
        
        # 8. Apply dynamic feature weighting
        x_trans = self.eca(x_trans)
        
        # 9. Global pooling across the time dimension: collapses to (B, 128)
        x_pooled = self.global_pool(x_trans).squeeze(-1)
        x_pooled = self.dropout(x_pooled)
        
        # 10. Output final class logits
        logits = self.classifier(x_pooled)
        
        return logits