import time
import json
import os
import socket
import sys
import numpy as np
import torch
import threading
from pynput.keyboard import Key, Controller

# Pathing setup to ensure it runs from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utility.helper import find_setting_in_directory, parse_full_frame
from utility.mmw_cube_proc_v2 import CubeProcessor  # v2 to match your Pi data
from model.one_d_tcn import GestureRecognitionNetwork

class InferenceEngine:
    def __init__(self, port, setting, model_path):
        self.port = port
        
        self.__mmw_proc = CubeProcessor(setting, mti_alpha=0.5)
        self.noise_threshold = 2.7
                
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GestureRecognitionNetwork(num_classes=6).to(self.device)
        
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

        self.classes = ["Hand Away", "Hand Towards", "Swipe Down", "Swipe Left", "Swipe Right", "Swipe Up"]
        self.frame_counter = 0

        # Asymmetric Hysteresis Configuration
        self.prediction_history = []
        self.class_rules = {
            "Hand Towards": {"frames": 6, "conf": 0.90},
            "Hand Away":    {"frames": 4, "conf": 0.80},
            "Swipe Left":   {"frames": 5, "conf": 0.85}, # Stricter to bypass chest interference
            "Swipe Right":  {"frames": 3, "conf": 0.80}, # Clean outward motion
            "Swipe Up":     {"frames": 4, "conf": 0.80},
            "Swipe Down":   {"frames": 4, "conf": 0.80}
        }
        
        self.keyboard = Controller()
        self.cooldown_frames = 0
        self.cooldown_threshold = 60

        self.os_control_active = False
        self.gui_callback = None

    def extract_rve_features(self, power_rdm, complex_cube, M=20):
        masked_rdm = power_rdm.copy()
        
        # Range Gating
        masked_rdm[:, 17:] = 0 
        masked_rdm[:8, :] = 0 
        masked_rdm[-8:, :] = 0 

        # Extract Top M Peaks
        flat_indices = np.argsort(masked_rdm.flatten())[-M:]
        d_idx, r_idx = np.unravel_index(flat_indices, masked_rdm.shape)
        
        r_vals, v_vals, az_vals, el_vals = [], [], [], []
        
        for i in range(len(r_idx)):
            d = d_idx[i]
            r = r_idx[i]
            
            r_vals.append(float(r))
            v_vals.append(float(d)) 
            
            rx_corner = complex_cube[d, r, 0]
            rx_az     = complex_cube[d, r, 1]
            rx_el     = complex_cube[d, r, 2]
            
            phase_az = np.angle(rx_corner * np.conj(rx_az))
            phase_el = np.angle(rx_corner * np.conj(rx_el))
            
            az_vals.append(np.arcsin(np.clip(phase_az / np.pi, -1.0, 1.0)))
            el_vals.append(np.arcsin(np.clip(phase_el / np.pi, -1.0, 1.0)))
            
        return (np.mean(r_vals), np.mean(v_vals), np.mean(az_vals), np.mean(el_vals))

    def execute_os_action(self, gesture):
        if gesture == "Swipe Left":
            for _ in range(4):
                self.keyboard.press(Key.media_volume_down)
                self.keyboard.release(Key.media_volume_down)
            # self.keyboard.press(Key.media_previous)
            # self.keyboard.release(Key.media_previous)
        elif gesture == "Swipe Right":
            for _ in range(4):
                self.keyboard.press(Key.media_volume_up)
                self.keyboard.release(Key.media_volume_up)
            # self.keyboard.press(Key.media_next)
            # self.keyboard.release(Key.media_next)
        elif gesture == "Swipe Up":
            self.keyboard.press(Key.media_next)
            self.keyboard.release(Key.media_next)
        elif gesture == "Swipe Down":
            self.keyboard.press(Key.media_previous)
            self.keyboard.release(Key.media_previous)
        elif gesture == "Hand Towards" or gesture == "Hand Away":
            self.keyboard.press(Key.media_play_pause)
            self.keyboard.release(Key.media_play_pause)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.port))
        sock.setblocking(False) 
        
        print(f"📥 Listening for UDP stream on port {self.port}...\n")
        print("--- Waiting for gesture ---")

        try:
            while True:
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
                
                # --- TIMING PHASE 1: Start & DSP ---
                start_time = time.time()

                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1
                    
                    if self.cooldown_frames == 0:
                        print("\n🟢 [READY] Cooldown finished. Listening for gestures...")

                (_, _, _, raw_payload) = parse_full_frame(latest_data)
                
                self.__mmw_proc.process_raw_data(list(raw_payload))
                
                curr_rdm = self.__mmw_proc.power_rdm
                complex_cube = self.__mmw_proc.complex_cube
                
                max_energy = np.max(np.log10(curr_rdm + 1e-9))
                
                # --- TIMING PHASE 2: Feature Extraction ---
                t_dsp = time.time() 
                
                r, v, a, e = self.extract_rve_features(curr_rdm, complex_cube, M=20)

                self.r_buf.append(r)
                self.v_buf.append(v)
                self.a_buf.append(a)
                self.e_buf.append(e)

                if len(self.r_buf) > 40:
                    self.r_buf.pop(0)
                    self.v_buf.pop(0)
                    self.a_buf.pop(0)
                    self.e_buf.pop(0)

                # --- TIMING PHASE 3: Inference ---
                t_feat = time.time()
                ran_inference = False

                if len(self.r_buf) == 40 and self.cooldown_frames == 0 and max_energy > self.noise_threshold:
                    ran_inference = True
                    
                    rt_raw = torch.FloatTensor(self.r_buf)
                    vt_raw = torch.FloatTensor(self.v_buf)
                    at_raw = torch.FloatTensor(self.a_buf)
                    et_raw = torch.FloatTensor(self.e_buf)

                    rt = rt_raw.view(1, 1, 40).to(self.device)
                    vt = vt_raw.view(1, 1, 40).to(self.device)
                    at = at_raw.view(1, 1, 40).to(self.device)
                    et = et_raw.view(1, 1, 40).to(self.device)

                    with torch.no_grad():
                        logits = self.model(rt, vt, at, et)
                        probs = torch.softmax(logits, dim=1)
                        conf, idx = torch.max(probs, dim=1)
                        gesture_name = self.classes[idx]
                        
                        if conf.item() > 0.65:
                            self.prediction_history.append(gesture_name)
                            
                            if len(self.prediction_history) > 10:
                                self.prediction_history.pop(0)

                            req_frames = self.class_rules[gesture_name]["frames"]
                            req_conf = self.class_rules[gesture_name]["conf"]

                            if conf.item() >= req_conf and len(self.prediction_history) >= req_frames:
                                recent_streak = self.prediction_history[-req_frames:]
                                
                                if all(g == gesture_name for g in recent_streak):
                                    
                                    print(f"🎯 VERIFIED: {gesture_name.ljust(15)} | Conf: {conf.item()*100:2.0f}% | Energy: {max_energy:.1f}")
                                    
                                    if self.os_control_active:
                                        threading.Thread(target=self.execute_os_action, args=(gesture_name,), daemon=True).start()
                                        print("   ⚡ Executed OS Action!")
                                    
                                    if self.gui_callback:
                                        self.gui_callback(gesture_name, conf.item()*100)
                                    
                                    self.cooldown_frames = self.cooldown_threshold
                                    self.prediction_history.clear()

                                    self.r_buf = [0.0] * 40
                                    self.v_buf = [16.0] * 40  
                                    self.a_buf = [0.0] * 40
                                    self.e_buf = [0.0] * 40
                        else:
                            self.prediction_history.clear()

                # --- LATENCY CALCULATIONS ---
                t_end = time.time()
                
                dsp_ms = (t_dsp - start_time) * 1000
                feat_ms = (t_feat - t_dsp) * 1000
                inf_ms = (t_end - t_feat) * 1000 if ran_inference else 0.0
                total_ms = (t_end - start_time) * 1000
                
                self.frame_counter += 1
                
                # Dropped heartbeat from 60 to 30 for ~1 second updates
                if self.frame_counter % 30 == 0:
                    inf_str = f"{inf_ms:.1f}" if ran_inference else "Skip"
                    print(f"[{self.frame_counter:05d}] Latency: {total_ms:>4.1f}ms | DSP: {dsp_ms:>4.1f}ms | Feat: {feat_ms:>4.1f}ms | Model: {inf_str:>4} | CD: {self.cooldown_frames}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Shutting down gracefully...")
        finally:
            sock.close()

def main():
    PORT = 9575
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    cfg_path = os.path.join(root_dir, "radar_config", "config_3rx_2m")
    model_path = os.path.join(root_dir, "weights_gacor", "best_fmcw_model_v51_ss_01.pth")
    
    setting_fn = find_setting_in_directory(cfg_path)
    with open(setting_fn, 'r') as f:
        setting = json.load(f)

    engine = InferenceEngine(PORT, setting, model_path)
    engine.run()

if __name__ == '__main__':
    main()