import torch
from thop import profile
from thop import clever_format
from model.one_d_tcn import GestureRecognitionNetwork

def profile_gesture_model():
    # 1. Instantiate the model
    # We use num_classes=12 to match your dataset
    model = GestureRecognitionNetwork(num_classes=12)
    
    # 2. Create dummy inputs matching the exact shape of a single inference
    # Shape: (Batch_Size=1, Channels=1, Sequence_Length=40)
    dummy_range = torch.randn(1, 1, 40)
    dummy_vel = torch.randn(1, 1, 40)
    dummy_az = torch.randn(1, 1, 40)
    
    # Pack inputs into a tuple for the thop profiler
    inputs = (dummy_range, dummy_vel, dummy_az)
    
    print("Profiling the model...")
    
    # 3. Calculate MACs (Multiply-Accumulate Operations) and Parameters
    # Note: 1 MAC is typically counted as 2 FLOPs in some literature, 
    # but thop calculates MACs and we usually report it directly or multiply by 2.
    macs, params = profile(model, inputs=inputs, verbose=False)
    
    # 4. Format the output to be easily readable (e.g., K for thousands, M for millions)
    formatted_macs, formatted_params = clever_format([macs, params], "%.2f")
    
    # 5. Standard PyTorch way to count trainable parameters (just to verify)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*50)
    print(f"Model Complexity Report")
    print("="*50)
    print(f"Total Parameters (thop):      {formatted_params}")
    print(f"Trainable Parameters (exact): {trainable_params:,}")
    print(f"MACs (Operations):            {formatted_macs}")
    
    # Estimate FLOPs (Usually 2 * MACs because one MAC = 1 multiply + 1 add)
    flops = macs * 2
    formatted_flops, _ = clever_format([flops, params], "%.2f")
    print(f"Estimated FLOPs:              {formatted_flops}")
    print("="*50)

if __name__ == "__main__":
    profile_gesture_model()