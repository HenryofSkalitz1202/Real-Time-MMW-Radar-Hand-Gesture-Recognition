import os, sys, json, glob
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the NEW v1 Cube Processor
from utility.mmw_cube_proc_v1 import CubeProcessor
from utility.helper import find_setting_in_directory

def save_to_research_csv(points, filename):
    df = pd.DataFrame(points, columns=['FrameNumber', 'ObjectNumber', 'Range', 'Velocity', 'PeakValue', 'Azimuth', 'Elevation'])
    df.to_csv(filename, index=False)
    print(f"Exported: {filename}")
    
def main(data_fn, cfg_dir):
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting = json.load(file)

    mmw_proc = CubeProcessor(setting, mti_alpha=0.5)

    all_points = []

    with open(data_fn, "rb") as file:
        while True:
            header = file.read(12)
            if len(header) < 12: break
            seq = int.from_bytes(header[4:8], 'little')
            data_len = int.from_bytes(header[8:12], 'little')
            
            # Process the raw binary frame
            mmw_proc.process_raw_data(file.read(data_len))

            # Fetch the 2D Power Map and 3D Phase Cube from v1 processor
            current_rdm = mmw_proc.power_rdm       # Shape: (Doppler, Range)
            complex_cube = mmw_proc.complex_cube   # Shape: (Doppler, Range, 3 Antennas)

            processed_rdm = current_rdm.copy()   

            # 2. RANGE GATING (Axis 1 is Range -> Kill ghosts past 51cm)
            processed_rdm[:, 17:] = 0
            
            # 3. DOPPLER MASKING (Axis 0 is Doppler -> Kill static clutter)
            processed_rdm[:8, :] = 0 
            processed_rdm[-8:, :] = 0
            
            # 5. Extract Top 20 Peaks
            flat_indices = np.argsort(processed_rdm.flatten())[-20:]
            d_idx, r_idx = np.unravel_index(flat_indices, processed_rdm.shape)

            for i in range(len(r_idx)):
                d = d_idx[i]
                r = r_idx[i]

                # 6. MONOPULSE PHASE EXTRACTION
                # Grab the raw complex numbers for this specific peak across the 3 antennas
                rx_corner = complex_cube[d, r, 0]
                rx_az     = complex_cube[d, r, 1]
                rx_el     = complex_cube[d, r, 2]
                
                # Calculate the continuous phase differences
                phase_az = np.angle(rx_corner * np.conj(rx_az))
                phase_el = np.angle(rx_corner * np.conj(rx_el))
                
                # Convert phase to radians (assuming standard lambda/2 spacing)
                theta = np.arcsin(np.clip(phase_az / np.pi, -1.0, 1.0))
                phi = np.arcsin(np.clip(phase_el / np.pi, -1.0, 1.0))

                # Hapus operasi Kartesian x, y, z. Simpan nilai langsung.
                all_points.append({
                    'FrameNumber': seq,
                    'ObjectNumber': i,
                    'Range': r,
                    'Velocity': d,
                    'PeakValue': processed_rdm[d, r], # Disimpan dalam skala linear
                    'Azimuth': theta,
                    'Elevation': phi
                })

    base_filename = os.path.basename(data_fn)
    csv_filename = base_filename.replace(".bin", ".csv")
    output_filepath = os.path.join("csv", csv_filename)
    
    save_to_research_csv(all_points, output_filepath)
    
if __name__ == '__main__':
    os.makedirs("csv", exist_ok=True)
    
    bin_files = glob.glob("data_v9/**/*.bin", recursive=True) 
    print(f"Starting batch process for {len(bin_files)} files...")

    for f in bin_files:
        main(f, "../radar_config/config_3rx_2m")
