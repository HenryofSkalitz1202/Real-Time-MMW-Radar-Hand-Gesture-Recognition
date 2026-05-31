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

from utility.helper import find_setting_in_directory, parse_full_frame

# ---> CHANGED: Pointing to the NEW v1 Cube Processor
from utility.mmw_cube_proc_v1 import CubeProcessor

from model.one_d_tcn import GestureRecognitionNetwork

# This order is required to avoid problems on Raspberry Pi.
app = QApplication(sys.argv)
import pyqtgraph as pg

class ImageGrid(QWidget):
    def __init__(self, M, N):
        super().__init__()
        self.M = M
        self.N = N
        self.initUI()

    def initUI(self):
        # Set up grid layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

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

        # Generate viridis colormap lookup table
        colormap = pg.colormap.get('viridis')
        lut = colormap.getLookupTable(0.0, 1.0, 256)

        self.image_items = []
        self.plot_items = []  
        for i in range(self.M):
            row_items = []
            row_plots = []
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

        self.setWindowTitle('Tugas Akhir: Live Monopulse Inference')
        self.resize(1280, 900)

    def update_prediction_text(self, text):
        self.result_label.setText(text)

    def set_plot_title_and_labels(self, row, col, title, x_name, y_name):
        axis_label_dict = {"Range": "Range(m)",
                           "Doppler": "Doppler(Hz)",
                           "Azimuth": "Azimuth(deg)",
                           "Elevation": "Elevation(deg)"}

        """Set the title and axis labels for a specific plot at (row, col)."""
        if 0 <= row < self.M and 0 <= col < self.N:
            plot_item = self.plot_items[row][col]
            plot_item.setTitle(title, size="18pt")
            bottom_axis = plot_item.getAxis('bottom')
            left_axis = plot_item.getAxis('left')
            label_font = pg.QtGui.QFont("Arial", 16)
            tick_font = pg.QtGui.QFont("Arial", 14)
            bottom_axis.setLabel(axis_label_dict[x_name])
            bottom_axis.label.setFont(label_font)
            bottom_axis.setStyle(tickFont=tick_font)
            left_axis.setLabel(axis_label_dict[y_name])
            left_axis.label.setFont(label_font)
            left_axis.setStyle(tickFont=tick_font)
            plot_item.showAxis('bottom')
            plot_item.showAxis('left')

    def display_image(self, row, col, image_data):
        if 0 <= row < self.M and 0 <= col < self.N:
            self.image_items[row][col].setImage(image_data)


class ImageUpdateThread(QThread):
    update_image_signal = pyqtSignal(int, int, np.ndarray)
    update_prediction_signal = pyqtSignal(str)

    def __init__(self, port, setting, plots, grid, model_path, parent=None, save_to_file=None):
        super().__init__(parent)
        self.__port = port
        self.__is_running = True
        self.__setting = setting
        self.__plots = plots
        self.__grid = grid  
        self.__save_to_file = save_to_file

        # ---> CHANGED: Initialized the new v1 Processor (no angle bins)
        self.__mmw_proc = CubeProcessor(setting)
        
        self.noise_threshold = 10.3
        
        # ML Inference Setup (4 Classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GestureRecognitionNetwork(num_classes=4).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ ERROR: Model weights not found at {model_path}")

        # Sequence Buffers
        self.r_buf, self.v_buf, self.a_buf, self.e_buf = [], [], [], []
        self.prev_rdm = None
        self.classes = ["Hand Away", "Hand Towards", "Swipe Left", "Swipe Right"]

    # ---> CHANGED: Replaced with the Monopulse Phase Logic
    def extract_rve_features(self, power_rdm, complex_cube, M=15):
        masked_rdm = power_rdm.copy()
        
        # 1. RANGE GATING (Axis 1)
        masked_rdm[:, 33:] = 0 

        # 2. DOPPLER MASKING (Axis 0)
        masked_rdm[:8, :] = 0 
        masked_rdm[-8:, :] = 0 

        masked_rdm = np.log10(masked_rdm + 1e-9)

        # 3. Extract top M peaks from the 2D Map
        flat_indices = np.argsort(masked_rdm.flatten())[-M:]
        d_idx, r_idx = np.unravel_index(flat_indices, masked_rdm.shape)
        
        r_vals, v_vals, az_vals, el_vals, weights = [], [], [], [], []
        
        for i in range(len(r_idx)):
            d = d_idx[i]
            r = r_idx[i]
            
            r_vals.append(r * 0.03)
            v_vals.append((d - 30) * 0.039 * 1.5) 
            
            # 4. MONOPULSE PHASE EXTRACTION
            rx_corner = complex_cube[d, r, 0]
            rx_az     = complex_cube[d, r, 1]
            rx_el     = complex_cube[d, r, 2]
            
            phase_az = np.angle(rx_corner * np.conj(rx_az))
            phase_el = np.angle(rx_corner * np.conj(rx_el))
            
            az_vals.append(np.arcsin(np.clip(phase_az / np.pi, -1.0, 1.0)))
            el_vals.append(np.arcsin(np.clip(phase_el / np.pi, -1.0, 1.0)))
            
            weights.append(masked_rdm[d, r])
            
        weights = np.array(weights) + 1e-9
        
        # 5. POWER-WEIGHTED CENTER OF MASS
        return (
            np.average(r_vals, weights=weights),
            np.average(v_vals, weights=weights),
            np.average(az_vals, weights=weights),
            np.average(el_vals, weights=weights)
        )

    def run(self):
        file_fd = None
        if self.__save_to_file is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename_with_timestamp = f"{self.__save_to_file}_{timestamp}.bin"
            directory = os.path.dirname(filename_with_timestamp)
            if directory: 
                os.makedirs(directory, exist_ok=True)
            file_fd = open(filename_with_timestamp, "wb")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.__port))
        
        while self.__is_running:
            try:
                raw_bytes, _ = sock.recvfrom(131072)
                if file_fd is not None:
                    file_fd.write(raw_bytes)
                (version, seq, data_len, raw_data) = parse_full_frame(raw_bytes)
                raw_data = list(raw_data)
                
                # Process the frame through the NEW v1 radar DSP pipeline
                self.__mmw_proc.process_raw_data(raw_data)
                
                # Fetch the 2D Power Map and 3D Phase Cube
                curr_rdm = self.__mmw_proc.power_rdm
                complex_cube = self.__mmw_proc.complex_cube

                if self.prev_rdm is not None:
                    diff_rdm = np.abs(curr_rdm - self.prev_rdm)
                    max_energy = np.max(np.log10(diff_rdm + 1e-9))
                    
                    if max_energy > self.noise_threshold:
                        # Extract physics
                        r, v, a, e = self.extract_rve_features(diff_rdm, complex_cube)
                    else:
                        r, v, a, e = 0.0, 0.0, 0.0, 0.0

                    self.r_buf.append(r)
                    self.v_buf.append(v)
                    self.a_buf.append(a)
                    self.e_buf.append(e)

                    if len(self.r_buf) > 40:
                        self.r_buf.pop(0)
                        self.v_buf.pop(0)
                        self.a_buf.pop(0)
                        self.e_buf.pop(0)

                    if len(self.r_buf) == 40:
                        rt_raw = torch.FloatTensor(self.r_buf)
                        vt_raw = torch.FloatTensor(self.v_buf)
                        at_raw = torch.FloatTensor(self.a_buf)
                        et_raw = torch.FloatTensor(self.e_buf)

                        # Identical Global Min-Max Scaling
                        rt = (rt_raw / 1.0).view(1, 1, 40).to(self.device)
                        vt = (vt_raw / 2.0).view(1, 1, 40).to(self.device)
                        at = (at_raw / 1.57).view(1, 1, 40).to(self.device)
                        et = (et_raw / 1.57).view(1, 1, 40).to(self.device)

                        with torch.no_grad():
                            logits = self.model(rt, vt, at, et)
                            probs = torch.softmax(logits, dim=1)
                            conf, idx = torch.max(probs, dim=1)
                            
                            if conf.item() > 0.80 and max_energy > self.noise_threshold:
                                gesture_name = self.classes[idx]
                                self.update_prediction_signal.emit(f"{gesture_name} ({conf.item()*100:.0f}%)")
                            else:
                                self.update_prediction_signal.emit("Scanning...")

                # Update history for the next MTI subtraction
                self.prev_rdm = curr_rdm
                
                # ---> CHANGED: UI Update (We only have a 2D map now, so we only update 1 plot!)
                img = self.__mmw_proc.vis_2d()
                self.update_image_signal.emit(0, 0, np.log10(img + 1e-9))
                
            except Exception as e:
                print(f"Error in run loop: {e}")
                continue

    def stop(self):
        self.__is_running = False
        self.quit()
        self.wait()

def main(port, cfg_dir, num_rows, num_cols, plots, model_path, fn=None):
    # Notice we only need a 1x1 grid now since we only output Range-Doppler!
    grid = ImageGrid(1, 1) 
    
    # Just setting up the single Range-Doppler plot
    grid.set_plot_title_and_labels(0, 0, "Range-Doppler Map", "Range", "Doppler")
    grid.show()
    
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting_data = json.load(file)
        
    thread = ImageUpdateThread(port, setting_data, plots, grid, model_path, save_to_file=fn)
    thread.update_image_signal.connect(grid.display_image)
    thread.update_prediction_signal.connect(grid.update_prediction_text)
    thread.start()
    
    try:
        sys.exit(app.exec_())
    finally:
        thread.stop()

if __name__ == '__main__':
    port = 9575
    
    # We only need 1 plot now
    num_rows = 1
    num_cols = 1
    plots = [(0, 0, "Range", "Doppler")]
    
    cfg_dir = "../radar_config/config_3rx_2m"
    model_path = "weights/best_fmcw_model_v6.pth"
    fn = "data/mmw_udp"
    
    main(port, cfg_dir, num_rows, num_cols, plots, model_path, fn)