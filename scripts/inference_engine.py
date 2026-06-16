import time
import json
import os
import socket
import sys
import numpy as np
import torch
from pynput.keyboard import Key, Controller  # OS Control Import

import threading

# Pathing setup to ensure it runs from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utility.helper import find_setting_in_directory, parse_full_frame
from utility.mmw_cube_proc_v1 import CubeProcessor
from model.one_d_tcn import GestureRecognitionNetwork

class InferenceEngine:
    def __init__(self, port, setting, model_path):
        self.port = port
        self.__mmw_proc = CubeProcessor(setting)
        self.noise_threshold = 4.5
                
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GestureRecognitionNetwork(num_classes=6).to(self.device)

        # --- Threading Flags ---
        self.is_processing_active = threading.Event()
        self.is_processing_active.clear() # Mulai dalam keadaan PAUSED (Sesuai GUI)
        self.shutdown_flag = threading.Event()
        self.debug_mode = True # Dikontrol oleh GUI
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Headless Mode Active. Model loaded from {model_path}")
        else:
            print(f"❌ ERROR: Model weights not found at {model_path}")
            sys.exit(1)

        self.r_buf = []
        self.v_buf = []
        self.a_buf = []
        self.e_buf = []
        self.prev_rdm = None

        self.classes = ["Hand Away", "Hand Towards", "Swipe Down", "Swipe Left", "Swipe Right", "Swipe Up"]
        self.frame_counter = 0

        # --- OS Control & Debouncing Setup ---
        self.keyboard = Controller()
        self.cooldown_frames = 0
        # Wait 20 frames (~0.6 seconds at 30fps) before allowing a new OS action.
        # Tune this up or down depending on your physical radar's frame rate.
        self.cooldown_threshold = 45

    def extract_rve_features(self, power_rdm, complex_cube, M=8):
        masked_rdm = power_rdm.copy()
        masked_rdm[:, 17:] = 0 
        masked_rdm[:8, :] = 0 
        masked_rdm[-8:, :] = 0 

        masked_rdm = np.log10(masked_rdm + 1e-9)

        flat_indices = np.argsort(masked_rdm.flatten())[-M:]
        d_idx, r_idx = np.unravel_index(flat_indices, masked_rdm.shape)
        
        r_vals, v_vals, az_vals, el_vals, weights = [], [], [], [], []
        
        for i in range(len(r_idx)):
            d = d_idx[i]
            r = r_idx[i]
            
            r_vals.append(r * 0.03)
            v_vals.append((d - 30) * 0.039 * 1.5) 
            
            rx_corner = complex_cube[d, r, 0]
            rx_az     = complex_cube[d, r, 1]
            rx_el     = complex_cube[d, r, 2]
            
            phase_az = np.angle(rx_corner * np.conj(rx_az))
            phase_el = np.angle(rx_corner * np.conj(rx_el))
            
            az_vals.append(np.arcsin(np.clip(phase_az / np.pi, -1.0, 1.0)))
            el_vals.append(np.arcsin(np.clip(phase_el / np.pi, -1.0, 1.0)))
            
            # CRITICAL FIX: Convert log weights back to linear!
            linear_power = 10 ** masked_rdm[d, r]
            weights.append(linear_power)
            
        weights = np.array(weights) + 1e-9
        
        return (
            np.average(r_vals, weights=weights),
            np.average(v_vals, weights=weights),
            np.average(az_vals, weights=weights),
            np.average(el_vals, weights=weights)
        )

    def execute_os_action(self, gesture):
        """Maps recognized gestures to OS media controls."""
        if gesture == "Swipe Left":
            # Previous Track
            self.keyboard.press(Key.media_previous)
            self.keyboard.release(Key.media_previous)
            
        elif gesture == "Swipe Right":
            # Next Track
            self.keyboard.press(Key.media_next)
            self.keyboard.release(Key.media_next)
            
        elif gesture == "Swipe Up":
            # Volume Up (Looping it 4 times makes the volume jump more noticeable per swipe)
            for _ in range(4):
                self.keyboard.press(Key.media_volume_up)
                self.keyboard.release(Key.media_volume_up)
                
        elif gesture == "Swipe Down":
            # Volume Down
            for _ in range(4):
                self.keyboard.press(Key.media_volume_down)
                self.keyboard.release(Key.media_volume_down)
                
        elif gesture == "Hand Towards" or gesture == "Hand Away":
            # Play / Pause Toggle
            self.keyboard.press(Key.media_play_pause)
            self.keyboard.release(Key.media_play_pause)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.port))
        sock.setblocking(False) 
        
        print(f"📥 Backend siap mendengarkan port {self.port}...")

        try:
            while not self.shutdown_flag.is_set():
                # 1. Kuras Buffer UDP (Harus selalu jalan agar jaringan tidak delay)
                latest_data = None
                while True:
                    try:
                        data, _ = sock.recvfrom(131072)
                        latest_data = data
                    except BlockingIOError:
                        break 
                        
                if latest_data is None:
                    time.sleep(0.005)
                    continue
                
                # 2. PAUSE PURGE LOGIC (Cegah halusinasi gestur saat resume)
                if not self.is_processing_active.is_set():
                    self.r_buf.clear()
                    self.v_buf.clear()
                    self.a_buf.clear()
                    self.e_buf.clear()
                    self.prev_rdm = None
                    continue # Lewati inferensi jika GUI sedang dalam status NONAKTIF

                # 3. AKTIF: Eksekusi DSP & PyTorch
                start_time = time.time()

                # Decrease cooldown counter every processed frame
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1

                (_, _, _, raw_payload) = parse_full_frame(latest_data)
                self.__mmw_proc.process_raw_data(list(raw_payload))

                curr_rdm = self.__mmw_proc.power_rdm
                complex_cube = self.__mmw_proc.complex_cube

                if self.prev_rdm is not None:
                    diff_rdm = np.abs(curr_rdm - self.prev_rdm)
                    max_energy = np.max(np.log10(diff_rdm + 1e-9))
                    
                    if max_energy > self.noise_threshold:
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

                    # --- PyTorch Inference ---
                    if len(self.r_buf) == 40 and sum(np.abs(self.v_buf)) != 0.0:
                        rt = (torch.FloatTensor(self.r_buf) / 1.0).view(1, 1, 40).to(self.device)
                        vt = (torch.FloatTensor(self.v_buf) / 2.0).view(1, 1, 40).to(self.device)
                        at = (torch.FloatTensor(self.a_buf) / 1.57).view(1, 1, 40).to(self.device)
                        et = (torch.FloatTensor(self.e_buf) / 1.57).view(1, 1, 40).to(self.device)

                        with torch.no_grad():
                            logits = self.model(rt, vt, at, et)
                            probs = torch.softmax(logits, dim=1)
                            conf, idx = torch.max(probs, dim=1)
                            gesture_name = self.classes[idx]
                            
                            # --- Output & Debouncing Logic ---
                            if max_energy > self.noise_threshold and conf.item() > 0.80:
                                if self.cooldown_frames == 0:
                                    if self.debug_mode:
                                        print(f"🎯 GESTUR: {gesture_name.ljust(15)} | Conf: {conf.item()*100:2.0f}% | Energi: {max_energy:.1f}")
                                        print(f"   ⚡ Mengeksekusi Tindakan OS! Cooldown dimulai.")
                                    
                                    self.execute_os_action(gesture_name)
                                    self.cooldown_frames = self.cooldown_threshold

                self.prev_rdm = curr_rdm

                # Heartbeat opsional
                if self.debug_mode:
                    self.frame_counter += 1
                    if self.frame_counter % 60 == 0:
                        proc_time = (time.time() - start_time) * 1000
                        print(f"[Heartbeat] Latensi Inferensi: {proc_time:.1f} ms | Sisa Cooldown: {self.cooldown_frames}")

        except Exception as e:
            print(f"Error pada thread inferensi: {e}")
        finally:
            sock.close()
            print("Socket ditutup.")