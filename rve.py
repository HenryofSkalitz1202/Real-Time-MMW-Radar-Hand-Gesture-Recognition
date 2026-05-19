import pandas as pd
import numpy as np

def process_csv_to_tensor(file_path, seq_length=40, M=7):
    """
    Reads a raw gesture CSV and converts it to a (3, seq_length) feature matrix.
    """
    try:
        # Load data and strip whitespace from headers
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # If the file is empty or corrupted, return a zero matrix
        if df.empty or 'Range' not in df.columns:
            return np.zeros((3, seq_length), dtype=np.float32)
            
        # 1. Calculate Azimuth
        safe_range = df['Range'].replace(0, 1e-6)
        sin_theta = np.clip(df['x'] / safe_range, -1.0, 1.0)
        df['Azimuth'] = np.arcsin(sin_theta)
        
        # 2. Extract Representative Values (Top M points per frame)
        def process_frame(frame_data):
            top_m = frame_data.sort_values(by='PeakValue', ascending=False).head(M)
            return pd.Series({
                'Range': top_m['Range'].mean(),
                'Velocity': top_m['Velocity'].mean(),
                'Azimuth': top_m['Azimuth'].mean()
            })
            
        frames = df.groupby('FrameNumber').apply(process_frame).reset_index(drop=True)
        
        # 3. Standardize Sequence Length (Pad with zeros or truncate to 40 frames)
        if len(frames) > seq_length:
            frames = frames.iloc[:seq_length]
        elif len(frames) < seq_length:
            padding = pd.DataFrame(0, index=np.arange(seq_length - len(frames)), columns=frames.columns)
            frames = pd.concat([frames, padding], ignore_index=True)
            
        # 4. Return as numpy array with shape (3 Features, 40 Frames)
        feature_matrix = frames[['Range', 'Velocity', 'Azimuth']].values.T
        return feature_matrix.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((3, seq_length), dtype=np.float32)