import torch
from thop import profile, clever_format
from model.lstm import LSTM_Gesture_Network

def profile_lstm_model():
    # 1. Instantiate the LSTM model
    # (Input: 3 features, Hidden: 32, Layers: 2, Classes: 12)
    model = LSTM_Gesture_Network(num_classes=12)
    
    # 2. Create dummy inputs (Batch=1, Channels=1, Sequence_Length=40)
    dummy_r = torch.randn(1, 1, 40)
    dummy_v = torch.randn(1, 1, 40)
    dummy_a = torch.randn(1, 1, 40)
    
    inputs = (dummy_r, dummy_v, dummy_a)
    
    print("Profiling the LSTM model...")
    
    # 3. Calculate MACs and Parameters
    macs, params = profile(model, inputs=inputs, verbose=False)
    
    # 4. Format the output
    formatted_macs, formatted_params = clever_format([macs, params], "%.2f")
    flops = macs * 2
    formatted_flops, _ = clever_format([flops, params], "%.2f")
    
    print("="*50)
    print("LSTM Model Complexity Report")
    print("="*50)
    print(f"Total Parameters: {formatted_params}")
    print(f"MACs (Ops):       {formatted_macs}")
    print(f"Estimated FLOPs:  {formatted_flops}")
    print("="*50)

if __name__ == "__main__":
    profile_lstm_model()