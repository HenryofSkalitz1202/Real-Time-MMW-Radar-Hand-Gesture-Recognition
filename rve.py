import pandas as pd
import numpy as np

def process_csv_to_tensor(file_path, seq_length=40, M=20):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Verifikasi ketersediaan kolom esensial
        if df.empty or 'Range' not in df.columns:
            return np.zeros((4, seq_length), dtype=np.float32)
        
        def process_frame(frame_data):
            top_m = frame_data.sort_values(by='PeakValue', ascending=False).head(M)
            
            r = top_m['Range'].mean()
            v = top_m['Velocity'].mean()
            az = top_m['Azimuth'].mean()
            el = top_m['Elevation'].mean()

            return pd.Series({'Range': r, 'Velocity': v, 'Azimuth': az, 'Elevation': el})
            
        frames = df.groupby('FrameNumber').apply(process_frame).reset_index(drop=True)
        
        if len(frames) > seq_length:
            frames = frames.iloc[:seq_length]
        elif len(frames) < seq_length:
            padding = pd.DataFrame(0.0, index=np.arange(seq_length - len(frames)), columns=frames.columns)
            
            padding['Velocity'] = 16.0
            frames = pd.concat([frames, padding], ignore_index=True)
            
        feature_matrix = frames[['Range', 'Velocity', 'Azimuth', 'Elevation']].values.T

        # Skalar normalisasi
        # feature_matrix[0, :] = feature_matrix[0, :] / 32.0                 # Range: dipetakan ke [0, 1]
        # feature_matrix[1, :] = (feature_matrix[1, :] - 16.0) / 16.0        # Velocity: dipetakan ke [-1, 1]
        # feature_matrix[2, :] = feature_matrix[2, :] / 1.57                 # Azimuth: dipetakan ke [-1, 1]
        # feature_matrix[3, :] = feature_matrix[3, :] / 1.57 

        return feature_matrix.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((4, seq_length), dtype=np.float32)