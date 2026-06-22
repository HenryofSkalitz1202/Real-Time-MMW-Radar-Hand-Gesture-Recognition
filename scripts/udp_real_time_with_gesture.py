import time
import json
import os
import socket
import sys
import numpy as np
import torch
from pynput.keyboard import Key, Controller  # OS Control Import

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
        self.cooldown_threshold = 45

        # --- GUI Hooks ---
        self.os_control_active = False  # Controlled by GUI toggle
        self.gui_callback = None        # Sends data back to GUI

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
        
        # Non-Blocking Mode for strict buffer draining
        sock.setblocking(False) 
        print(f"📥 Listening for UDP stream on port {self.port}...\n")
        print("--- Waiting for gesture ---")

        try:
            while True:
                # Buffer Drain Logic
                latest_data = None
                while True:
                    try:
                        data, _ = sock.recvfrom(131072)
                        latest_data = data
                    except BlockingIOError:
                        break 
                        
                # Sleep briefly if no packets arrived to prevent CPU maxing
                if latest_data is None:
                    time.sleep(0.005) # 5 milliseconds
                    continue
                
                # ⏱️ START TIMER
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

                    # Continuous Inference
                    if len(self.r_buf) == 40:
                        
                        # THE SILENCE GATE
                        if sum(np.abs(self.v_buf)) != 0.0:
                            rt_raw = torch.FloatTensor(self.r_buf)
                            vt_raw = torch.FloatTensor(self.v_buf)
                            at_raw = torch.FloatTensor(self.a_buf)
                            et_raw = torch.FloatTensor(self.e_buf)

                            rt = (rt_raw / 1.0).view(1, 1, 40).to(self.device)
                            vt = (vt_raw / 2.0).view(1, 1, 40).to(self.device)
                            at = (at_raw / 1.57).view(1, 1, 40).to(self.device)
                            et = (et_raw / 1.57).view(1, 1, 40).to(self.device)

                            with torch.no_grad():
                                logits = self.model(rt, vt, at, et)
                                probs = torch.softmax(logits, dim=1)
                                conf, idx = torch.max(probs, dim=1)
                                gesture_name = self.classes[idx]
                                
                                # Terminal Output & OS Action Logic (GUI Integrated)
                                if max_energy > self.noise_threshold:
                                    if conf.item() > 0.80:
                                        
                                        # Trigger logic only if cooldown is zero
                                        if self.cooldown_frames == 0:
                                            # 1. Print to Terminal
                                            print(f"🎯 GESTURE: {gesture_name.ljust(15)} | Confidence: {conf.item()*100:2.0f}% | Energy: {max_energy:.1f}")
                                            
                                            # 2. Trigger OS Action if enabled via GUI
                                            if self.os_control_active:
                                                self.execute_os_action(gesture_name)
                                                print("   ⚡ Executed OS Action!")
                                            
                                            # 3. Send to GUI if connected
                                            if self.gui_callback:
                                                self.gui_callback(gesture_name, conf.item()*100)
                                            
                                            self.cooldown_frames = self.cooldown_threshold # Reset cooldown

                self.prev_rdm = curr_rdm

                # ⏱️ STOP TIMER & PROFILE
                end_time = time.time()
                processing_time_ms = (end_time - start_time) * 1000
                
                self.frame_counter += 1
                if self.frame_counter % 60 == 0:
                    # Print a subtle heartbeat every ~2 seconds to prove the script hasn't frozen
                    print(f"[System Heartbeat] Math & Inference Latency: {processing_time_ms:.1f} ms | Cooldown: {self.cooldown_frames}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Shutting down gracefully...")
        finally:
            sock.close()

def main():
    PORT = 9575
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    cfg_path = os.path.join(root_dir, "radar_config", "config_3rx_2m")
    model_path = os.path.join(root_dir, "weights", "best_fmcw_model_v51_b32.pth")
    
    setting_fn = find_setting_in_directory(cfg_path)
    with open(setting_fn, 'r') as f:
        setting = json.load(f)

    engine = InferenceEngine(PORT, setting, model_path)
    engine.run()

if __name__ == '__main__':
    main()