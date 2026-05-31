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

# --- 2. The Complete SRDST Architecture ---
class SRDST_Adapted_Network(nn.Module):
    def __init__(self, num_classes=4, seq_len=40, num_channels=4, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # ==========================================
        # STREAM 1: Time Dimension Encoder
        # Extracts temporal correlations across frames
        # ==========================================
        self.time_embedding = nn.Linear(num_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.time_transformer = nn.TransformerEncoder(time_encoder_layer, num_layers=num_layers)
        
        # ==========================================
        # STREAM 2: Channel Dimension Encoder
        # Extracts correlations between physical variables
        # ==========================================
        self.channel_embedding = nn.Linear(seq_len, d_model)
        # No positional encoding is used here as per Jin et al.
        
        channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.channel_transformer = nn.TransformerEncoder(channel_encoder_layer, num_layers=num_layers)
        
        # ==========================================
        # FUSION LAYER MODULE
        # ==========================================
        # Flattened dimensions
        self.flat_time_dim = seq_len * d_model         # 40 * 32 = 1280
        self.flat_channel_dim = num_channels * d_model # 4 * 32 = 128
        self.concat_dim = self.flat_time_dim + self.flat_channel_dim # 1408
        
        # Weights generation network: (wc, wt) = FC[Concat(Flatten(Y_c), Flatten(Y_t))]
        self.weight_fc = nn.Sequential(
            nn.Linear(self.concat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1) # Yields two weights that sum to 1
        )
        
        # Final Classification Network
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.concat_dim, num_classes)
        )

    def forward(self, range_seq, vel_seq, az_seq, el_seq):
        # 1. Combine 4 features -> Shape: [batch, 4, 40]
        x = torch.cat([range_seq, vel_seq, az_seq, el_seq], dim=1)
        
        # ---------------------------------------------
        # Branch A: Time Dimension
        # ---------------------------------------------
        # Transpose to [batch, 40, 4] so sequence length is 40
        x_time = x.transpose(1, 2)
        
        y_t = self.time_embedding(x_time)
        y_t = self.pos_encoder(y_t)
        y_t = self.time_transformer(y_t)
        y_t_flat = torch.flatten(y_t, start_dim=1) # Shape: [batch, 1280]
        
        # ---------------------------------------------
        # Branch B: Channel Dimension
        # ---------------------------------------------
        # Keep as [batch, 4, 40] so sequence length is 4 (channels)
        x_channel = x
        
        y_c = self.channel_embedding(x_channel)
        # (No positional encoding)
        y_c = self.channel_transformer(y_c)
        y_c_flat = torch.flatten(y_c, start_dim=1) # Shape: [batch, 128]
        
        # ---------------------------------------------
        # Weighted Fusion
        # ---------------------------------------------
        # Concatenate for weight calculation
        concat_flat = torch.cat([y_c_flat, y_t_flat], dim=1) # Shape: [batch, 1408]
        
        # Generate weights
        weights = self.weight_fc(concat_flat)
        w_c = weights[:, 0].unsqueeze(1) # Shape: [batch, 1]
        w_t = weights[:, 1].unsqueeze(1) # Shape: [batch, 1]
        
        # Apply weights to the respective flattened feature vectors
        weighted_y_c = y_c_flat * w_c
        weighted_y_t = y_t_flat * w_t
        
        # ---------------------------------------------
        # Classification
        # ---------------------------------------------
        # Re-concatenate the weighted features
        weighted_concat = torch.cat([weighted_y_c, weighted_y_t], dim=1)
        
        logits = self.classifier(weighted_concat)
        return logits

# --- Quick Parameter Check (Optional) ---
if __name__ == "__main__":
    from thop import profile, clever_format
    model = SRDST_Adapted_Network(num_classes=4)
    
    # Dummy inputs representing DataLoader outputs [batch_size, 1, 40]
    dummy_r = torch.randn(1, 1, 40)
    dummy_v = torch.randn(1, 1, 40)
    dummy_a = torch.randn(1, 1, 40)
    dummy_e = torch.randn(1, 1, 40)
    
    macs, params = profile(model, inputs=(dummy_r, dummy_v, dummy_a, dummy_e), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    
    print(f"SRDST Adapted Parameters: {params}")
    print(f"SRDST Adapted MACs:       {macs}")