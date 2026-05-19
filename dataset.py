import os
import torch
from torch.utils.data import Dataset
from rve import process_csv_to_tensor

class RadarGestureDataset(Dataset):
    def __init__(self, data_dir="Data", seq_length=40):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.samples = []
        
        # Define all the gesture classes based on your folder names
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print("Scanning directory and building dataset...")
        # Loop through each folder and grab all CSV file paths
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            label = self.class_to_idx[cls_name]
            
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(cls_dir, file_name)
                    self.samples.append((file_path, label))
                    
        print(f"Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Call the RVE script to get the (3, 40) matrix
        feature_matrix = process_csv_to_tensor(file_path, self.seq_length)
        
        # Convert to PyTorch Tensors
        # Split the matrix into 3 separate branches (Range, Velocity, Azimuth)
        range_seq = torch.tensor(feature_matrix[0:1, :])    # Shape: (1, 40)
        vel_seq = torch.tensor(feature_matrix[1:2, :])      # Shape: (1, 40)
        az_seq = torch.tensor(feature_matrix[2:3, :])       # Shape: (1, 40)
        
        return range_seq, vel_seq, az_seq, torch.tensor(label, dtype=torch.long)