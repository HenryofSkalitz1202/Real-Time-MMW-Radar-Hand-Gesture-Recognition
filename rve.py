import pandas as pd
import numpy as np

## RVE: Representative Value Extraction

def process_csv_to_tensor(file_path, seq_length=40, M=20):
    """
    Reads a raw gesture CSV and converts it to a (4, seq_length) feature matrix.
    Output Order: [Range, Velocity, Azimuth, Elevation]
    """
    try:
        # Load data and strip whitespace from headers
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # If the file is empty or corrupted, return a zero matrix with 4 channels
        if df.empty or 'Range' not in df.columns:
            return np.zeros((4, seq_length), dtype=np.float32)
            
        # 1. Calculate Azimuth and Elevation
        safe_range = df['Range'].replace(0, 1e-6) # Prevent division by zero
        
        # Azimuth calculation
        sin_theta = np.clip(df['x'] / safe_range, -1.0, 1.0)
        df['Azimuth'] = np.arcsin(sin_theta)
        
        # Elevation calculation (z = R * sin(phi))
        # Adding a safety check in case you accidentally load an old CSV without the 'z' column
        if 'z' in df.columns:
            sin_phi = np.clip(df['z'] / safe_range, -1.0, 1.0)
            df['Elevation'] = np.arcsin(sin_phi)
        else:
            df['Elevation'] = 0.0
        
        # 2. Extract Representative Values (Top M points per frame)
        def process_frame(frame_data):
            top_m = frame_data.sort_values(by='PeakValue', ascending=False).head(M)

            # Extract the raw power values to use as gravity/weights
            weights = top_m['PeakValue'].values + 1e-9 # Add epsilon to prevent div by zero

            return pd.Series({
                # Calculate the Weighted Average (Center of Mass)
                'Range': np.average(top_m['Range'], weights=weights),
                'Velocity': np.average(top_m['Velocity'], weights=weights),
                'Azimuth': np.average(top_m['Azimuth'], weights=weights),
                'Elevation': np.average(top_m['Elevation'], weights=weights)
            })
            
        frames = df.groupby('FrameNumber').apply(process_frame).reset_index(drop=True)
        
        # 3. Standardize Sequence Length (Pad with zeros or truncate to 40 frames)
        if len(frames) > seq_length:
            frames = frames.iloc[:seq_length]
        elif len(frames) < seq_length:
            padding = pd.DataFrame(0, index=np.arange(seq_length - len(frames)), columns=frames.columns)
            frames = pd.concat([frames, padding], ignore_index=True)
            
        # 4. Return as numpy array with shape (4 Features, 40 Frames)
        feature_matrix = frames[['Range', 'Velocity', 'Azimuth', 'Elevation']].values.T

        # --- GLOBAL MIN-MAX SCALING ---
        # Preserves physical aspect ratios and absolute gesture sizes
        # feature_matrix[0, :] = feature_matrix[0, :] / 1.0   # Range (0 to 1.0m)
        # feature_matrix[1, :] = feature_matrix[1, :] / 2.0   # Velocity (+/- 2.0 m/s)
        # feature_matrix[2, :] = feature_matrix[2, :] / 1.57  # Azimuth (+/- pi/2 rads)
        # feature_matrix[3, :] = feature_matrix[3, :] / 1.57  # Elevation (+/- pi/2 rads

        return feature_matrix.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((4, seq_length), dtype=np.float32)