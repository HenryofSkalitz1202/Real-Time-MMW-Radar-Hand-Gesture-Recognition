import pandas as pd
import numpy as np

def process_csv_to_tensor(file_path, seq_length=40, M=20):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        if df.empty or 'Range' not in df.columns:
            return np.zeros((4, seq_length), dtype=np.float32)
            
        safe_range = df['Range'].replace(0, 1e-6) 
        
        # Pre-calculate Angles per row (Matching real-time script)
        #df['Azimuth'] = np.arcsin(np.clip(df['x'] / safe_range, -1.0, 1.0))

        # --- CORRECTED ANGLE CALCULATION ---
        # arctan2 takes (numerator, denominator) which perfectly isolates the horizontal angle
        df['Azimuth'] = np.arctan2(df['x'], df['y'])

        if 'z' in df.columns:
            df['Elevation'] = np.arcsin(np.clip(df['z'] / safe_range, -1.0, 1.0))
        else:
            df['Elevation'] = 0.0
        
        def process_frame(frame_data):
            # 1. THE SILENCE GATE (Match real-time threshold of 4.3)
            if frame_data['PeakValue'].max() < 4.3:
                return pd.Series({'Range': 0.0, 'Velocity': 0.0, 'Azimuth': 0.0, 'Elevation': 0.0})

            top_m = frame_data.sort_values(by='PeakValue', ascending=False).head(M)
            
            # 2. LINEAR WEIGHTS (Undo log10 to accurately track the hand)
            linear_weights = 10 ** (top_m['PeakValue'].values)
            weights = linear_weights + 1e-9 

            # 3. DIRECT ANGLE AVERAGING
            r = np.average(top_m['Range'], weights=weights)
            v = np.average(top_m['Velocity'], weights=weights)
            az = np.average(top_m['Azimuth'], weights=weights)
            el = np.average(top_m['Elevation'], weights=weights)

            return pd.Series({'Range': r, 'Velocity': v, 'Azimuth': az, 'Elevation': el})
            
        frames = df.groupby('FrameNumber').apply(process_frame).reset_index(drop=True)
        
        if len(frames) > seq_length:
            frames = frames.iloc[:seq_length]
        elif len(frames) < seq_length:
            padding = pd.DataFrame(0, index=np.arange(seq_length - len(frames)), columns=frames.columns)
            frames = pd.concat([frames, padding], ignore_index=True)
            
        feature_matrix = frames[['Range', 'Velocity', 'Azimuth', 'Elevation']].values.T

        # GLOBAL MIN-MAX SCALING
        feature_matrix[0, :] = feature_matrix[0, :] / 1.0   
        feature_matrix[1, :] = feature_matrix[1, :] / 2.0   
        feature_matrix[2, :] = feature_matrix[2, :] / 1.57  
        feature_matrix[3, :] = feature_matrix[3, :] / 1.57  

        return feature_matrix.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((4, seq_length), dtype=np.float32)