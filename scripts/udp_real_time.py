import datetime
import json
import os
import socket
import sys
import numpy as np
import torch

from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utility.helper import find_setting_in_directory, parse_full_frame
from utility.mmw_cube_proc_v0 import CubeProcessor

from model.one_d_tcn import GestureRecognitionNetwork

# Initialize the PyQt Application
app = QApplication(sys.argv)

class ImageGrid(QWidget):
    def __init__(self, M, N):
        super().__init__()
        self.M, self.N = M, N
        self.initUI()

    def initUI(self):
        # Main Vertical Layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Plot Container
        grid_container = QWidget()
        self.grid = QGridLayout()
        grid_container.setLayout(self.grid)
        self.main_layout.addWidget(grid_container)

        # High-Visibility Prediction Label
        self.result_label = QLabel("INITIALIZING SENSOR...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 44, QFont.Bold))
        self.result_label.setStyleSheet("""
            color: #2ecc71; 
            padding: 25px; 
            background-color: #2c3e50; 
            border-radius: 15px;
            margin-top: 10px;
        """)
        self.main_layout.addWidget(self.result_label)

        # Colormap setup
        colormap = pg.colormap.get('viridis')
        lut = colormap.getLookupTable(0.0, 1.0, 256)

        self.image_items = []
        self.plot_items = []  
        for i in range(self.M):
            row_items, row_plots = [], []
            for j in range(self.N):
                plot_item = pg.PlotItem()
                plot_item.setAspectLocked(True)
                plot_item.hideAxis('left')
                plot_item.hideAxis('bottom')
                
                image_item = pg.ImageItem()
                image_item.setLookupTable(lut)
                plot_item.addItem(image_item)
                
                view = pg.GraphicsLayoutWidget()
                view.addItem(plot_item)
                self.grid.addWidget(view, i, j)
                
                row_items.append(image_item)
                row_plots.append(plot_item)
            self.image_items.append(row_items)
            self.plot_items.append(row_plots)

        self.setWindowTitle('Tugas Akhir: Real-Time Gesture Inference')
        self.resize(1280, 900)

    def update_prediction_text(self, text):
        self.result_label.setText(text)

    def display_image(self, row, col, image_data):
        self.image_items[row][col].setImage(image_data)

    def set_plot_info(self, row, col, title):
        if 0 <= row < self.M and 0 <= col < self.N:
            self.plot_items[row][col].setTitle(title, size="14pt")

class ImageUpdateThread(QThread):
    update_image_signal = pyqtSignal(int, int, np.ndarray)
    update_prediction_signal = pyqtSignal(str)

    def __init__(self, port, setting, plots, model_path):
        super().__init__()
        self.__port = port
        self.__is_running = True
        self.__plots = plots
        self.__mmw_proc = CubeProcessor(setting, num_azimuth_bin=16, num_elevation_bin=16)
        
        # Set threshold high enough to ignore baseline room noise
        self.noise_threshold = 10.3
                
        # ML Inference Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ---> CHANGED: Updated num_classes to 4 to match the new pivot
        self.model = GestureRecognitionNetwork(num_classes=4).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ ERROR: Model weights not found at {model_path}")

        # Buffers for the 4 TCN branches
        self.r_buf, self.v_buf, self.a_buf, self.e_buf = [], [], [], []
        self.prev_cube = None

        # ---> CHANGED: Updated classes list to alphabetized 4-gesture subset
        self.classes = ["Hand Away", "Hand Towards", "Swipe Left", "Swipe Right"]

    # ---> CHANGED: M expanded to 15 for wider spatial footprint
    def extract_rve_features(self, cube, M=15):
        masked_cube = cube.copy()
        
        # 1. RANGE GATING (3cm resolution)
        # Zeros out everything past 1.0 meter (Bin 33)
        masked_cube[33:, :, :, :] = 0 

        # 2. Doppler Masking (Removes static clutter)
        masked_cube[:, :8, :, :] = 0 
        masked_cube[:, -8:, :, :] = 0 

        # 3. Apply Log Scaling (CRITICAL for matching AWR1642 magnitudes)
        masked_cube = np.log10(masked_cube + 1e-9)

        # 4. Extract the top M peaks from the masked, log-scaled data
        flat_indices = np.argsort(masked_cube.flatten())[-M:]
        r_idx, d_idx, az_idx, el_idx = np.unravel_index(flat_indices, masked_cube.shape)
        
        # 5. DOMAIN HACK MULTIPLIERS (BGT60TR13C 2m Config to AWR1642 approximation)
        r_vals = r_idx * 0.03 
        v_vals = (d_idx - 30) * 0.039 * 1.5 
        az_vals = (az_idx / 15.0) * (np.pi * 120 / 180) - (np.pi * 60 / 180)
        el_vals = (el_idx / 15.0) * (np.pi * 120 / 180) - (np.pi * 60 / 180)
        
        return np.mean(r_vals), np.mean(v_vals), np.mean(az_vals), np.mean(el_vals)

    def run(self):
        # 1. Setup Socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.__port))
        print(f"📥 Listening on port {self.__port}...")

        while self.__is_running:
            try:
                # Receive raw radar packet
                data, _ = sock.recvfrom(131072)
                (_, _, _, raw_payload) = parse_full_frame(data)
                
                # Process the frame through the radar DSP pipeline
                self.__mmw_proc.process_raw_data(list(raw_payload))
                curr_cube = np.abs(self.__mmw_proc.data_cube_fft)

                if self.prev_cube is not None:
                    diff_cube = np.abs(curr_cube - self.prev_cube)
                    max_energy = np.max(np.log10(diff_cube + 1e-9))
                    
                    if max_energy > self.noise_threshold:
                        # ACTIVE MOTION: Extract physical features
                        r, v, a, e = self.extract_rve_features(diff_cube)
                    else:
                        # STILLNESS: Feed zeros to the model to represent "resting" state
                        r, v, a, e = 0.0, 0.0, 0.0, 0.0

                    # Buffer Management (Sliding window of 40 frames)
                    self.r_buf.append(r)
                    self.v_buf.append(v)
                    self.a_buf.append(a)
                    self.e_buf.append(e)

                    if len(self.r_buf) > 40:
                        self.r_buf.pop(0)
                        self.v_buf.pop(0)
                        self.a_buf.pop(0)
                        self.e_buf.pop(0)

                    # Inference Block (When buffer is full)
                    if len(self.r_buf) == 40:
                        # Convert to tensors first
                        rt_raw = torch.FloatTensor(self.r_buf)
                        vt_raw = torch.FloatTensor(self.v_buf)
                        at_raw = torch.FloatTensor(self.a_buf)
                        et_raw = torch.FloatTensor(self.e_buf)

                        # ---> CHANGED: Apply identical Global Min-Max Scaling
                        rt = (rt_raw / 1.0).view(1, 1, 40).to(self.device)
                        vt = (vt_raw / 2.0).view(1, 1, 40).to(self.device)
                        at = (at_raw / 1.57).view(1, 1, 40).to(self.device)
                        et = (et_raw / 1.57).view(1, 1, 40).to(self.device)

                        # DEBUG Console output
                        if max_energy > self.noise_threshold:
                            print(f"MOTION detected! MaxE: {max_energy:.2f} | R: {r:.2f} | V: {v:.2f}")
                        else:
                            if self.r_buf[-1] == 0.0 and self.r_buf[-2] != 0.0:
                                print("STILLNESS: Noise floor reached.")

                        with torch.no_grad():
                            logits = self.model(rt, vt, at, et)
                            probs = torch.softmax(logits, dim=1)
                            conf, idx = torch.max(probs, dim=1)
                            
                            # E. UI Signal Update
                            # Condition: High confidence AND current frame has movement
                            if conf.item() > 0.80 and max_energy > self.noise_threshold:
                                gesture_name = self.classes[idx]
                                self.update_prediction_signal.emit(f"{gesture_name} ({conf.item()*100:.0f}%)")
                            else:
                                self.update_prediction_signal.emit("Scanning...")

                    # F. UI Plot Updates (Visualization)
                    for plot in self.__plots:
                        row, col, ax0, ax1 = plot
                        img = self.__mmw_proc.vis_2d(ax0, ax1)
                        # np.log10 ensures weak signals are still visible to the eye
                        self.update_image_signal.emit(row, col, np.log10(img + 1e-9))

                # Update history for the next MTI subtraction
                self.prev_cube = curr_cube
                
            except Exception as e:
                print(f"Error in run loop: {e}")
                continue

        sock.close()

    def stop(self):
        self.__is_running = False
        self.quit()
        self.wait()

def main():
    # Parameters
    PORT = 9575
    CFG_DIR = "radar_config/config_3rx_2m"
    
    # Make sure this points to the newly saved 4-class weights file!
    MODEL_PATH = "weights/best_fmcw_model_v6.pth" 
    
    PLOTS = [(0, 0, "Range", "Doppler"), (0, 1, "Azimuth", "Range"), (0, 2, "Azimuth", "Doppler")]

    # UI Setup
    grid = ImageGrid(1, 3)
    for plot in PLOTS:
        grid.set_plot_info(plot[0], plot[1], f"{plot[2]}-{plot[3]}")
    grid.show()

    # Load Radar Settings
    setting_fn = find_setting_in_directory(CFG_DIR)
    with open(setting_fn, 'r') as f:
        setting = json.load(f)

    # Thread Setup
    thread = ImageUpdateThread(PORT, setting, PLOTS, MODEL_PATH)
    thread.update_image_signal.connect(grid.display_image)
    thread.update_prediction_signal.connect(grid.update_prediction_text)
    thread.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()