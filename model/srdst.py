import torch
import torch.nn as nn
import math

# --- 1. Transformer Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Injects information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, max_len=40):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: [1, max_len, d_model] to broadcast across batches
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# --- 2. Single Transformer Stream ---
class TransformerStream(nn.Module):
    """A single branch of the Dual-Stream architecture."""
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        # Project our small input dimensions (1 or 2) up to the Transformer's working dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Standard PyTorch Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, src):
        # 1. Project to d_model space
        x = self.input_proj(src)
        # 2. Add positional context
        x = self.pos_encoder(x)
        # 3. Pass through self-attention layers
        x = self.transformer_encoder(x)
        return x

# --- 3. The Complete SRDST-Adapted Architecture ---
class SRDST_Adapted_Network(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        
        # Stream 1: Spatial Trajectory (Range, Azimuth -> 2 features)
        self.spatial_stream = TransformerStream(input_dim=2, d_model=32, nhead=4, num_layers=2)
        
        # Stream 2: Motion Dynamics (Velocity -> 1 feature)
        self.motion_stream = TransformerStream(input_dim=1, d_model=32, nhead=4, num_layers=2)
        
        # Feature Fusion & Classification
        # 32 dims from spatial + 32 dims from motion = 64
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, range_seq, vel_seq, az_seq):
        """
        Input shapes from your DataLoader: [batch_size, 1, 40]
        Transformers expect: [batch_size, seq_len, features]
        """
        # 1. Prepare and reshape inputs for the Transformers
        # Spatial: Combine Range and Azimuth, then swap dimensions -> [batch_size, 40, 2]
        spatial_in = torch.cat([range_seq, az_seq], dim=1).transpose(1, 2)
        
        # Motion: Just swap dimensions -> [batch_size, 40, 1]
        motion_in = vel_seq.transpose(1, 2)
        
        # 2. Pass through Dual-Stream Transformers
        out_spatial = self.spatial_stream(spatial_in) # Shape: [batch, 40, 32]
        out_motion = self.motion_stream(motion_in)    # Shape: [batch, 40, 32]
        
        # 3. Global Average Pooling (Collapse the 40 frames into a single rich feature vector)
        pooled_spatial = out_spatial.mean(dim=1) # Shape: [batch, 32]
        pooled_motion = out_motion.mean(dim=1)   # Shape: [batch, 32]
        
        # 4. Concatenate streams
        fused_features = torch.cat((pooled_spatial, pooled_motion), dim=1) # Shape: [batch, 64]
        
        # 5. Classification
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits

# --- Quick Parameter Check (Optional) ---
if __name__ == "__main__":
    from thop import profile, clever_format
    model = SRDST_Adapted_Network(num_classes=12)
    
    # Dummy inputs representing your DataLoader outputs
    dummy_r = torch.randn(1, 1, 40)
    dummy_v = torch.randn(1, 1, 40)
    dummy_a = torch.randn(1, 1, 40)
    
    macs, params = profile(model, inputs=(dummy_r, dummy_v, dummy_a), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    
    print(f"SRDST Adapted Parameters: {params}")
    print(f"SRDST Adapted MACs:       {macs}")