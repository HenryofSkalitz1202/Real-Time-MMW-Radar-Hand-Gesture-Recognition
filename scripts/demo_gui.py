import torch  # <-- Keep this at the absolute top to prevent WinError 1114!

import sys
import os
import json
import threading
import queue
import customtkinter as ctk

from udp_real_time_with_gesture import InferenceEngine, find_setting_in_directory

class InferenceThread(threading.Thread):
    """
    Runs the blocking inference loop in a background thread.
    Uses a Queue to safely pass data back to the CustomTkinter main thread.
    """
    def __init__(self, data_queue):
        # daemon=True ensures this thread closes automatically when you close the GUI
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.engine = None

    def run(self):
        PORT = 9575
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
        cfg_path = os.path.join(root_dir, "radar_config", "config_3rx_2m")
        model_path = os.path.join(root_dir, "weights", "best_fmcw_model_v51_b32.pth")
        
        setting_fn = find_setting_in_directory(cfg_path)
        with open(setting_fn, 'r') as f:
            setting = json.load(f)

        self.engine = InferenceEngine(PORT, setting, model_path)
        
        # Link the engine's callback to our queue
        self.engine.gui_callback = self.send_to_gui
        
        # Start the blocking loop
        self.engine.run()

    def send_to_gui(self, gesture, confidence):
        # Drop the detected gesture into the queue for the GUI to pick up
        self.data_queue.put((gesture, confidence))

    def toggle_os_control(self, is_active):
        if self.engine:
            self.engine.os_control_active = is_active


class GestureGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Radar Gesture Control")
        self.geometry("400x300")
        self.resizable(False, False)

        # Apply Global Minimalist Black Theme
        ctk.set_appearance_mode("dark")
        self.configure(fg_color="#0A0A0A")

        # Fonts (Plus Jakarta Sans)
        self.font_title = ctk.CTkFont(family="Plus Jakarta Sans", size=16, weight="bold")
        self.font_gesture = ctk.CTkFont(family="Plus Jakarta Sans", size=28, weight="bold")
        self.font_btn = ctk.CTkFont(family="Plus Jakarta Sans", size=12, weight="bold")

        self.init_ui()

        # Threading Setup
        self.data_queue = queue.Queue()
        self.worker = InferenceThread(self.data_queue)
        self.worker.start()

        # Start the queue polling loop
        self.check_queue()

    def init_ui(self):
        # Title Label
        self.title_label = ctk.CTkLabel(
            self, 
            text="GESTURE ENGINE", 
            font=self.font_title, 
            text_color="#FFFFFF"
        )
        self.title_label.pack(pady=(35, 10))

        # Separator Line
        self.separator = ctk.CTkFrame(self, height=2, fg_color="#333333", corner_radius=0)
        self.separator.pack(fill="x", padx=50, pady=(0, 40))

        # Gesture Display Label
        self.gesture_label = ctk.CTkLabel(
            self, 
            text="WAITING...", 
            font=self.font_gesture, 
            text_color="#FFFFFF"
        )
        self.gesture_label.pack(pady=(0, 40))

        # OS Action Toggle Button
        self.os_active = False
        self.toggle_btn = ctk.CTkButton(
            self,
            text="ENABLE OS ACTIONS",
            font=self.font_btn,
            width=200,
            height=40,
            corner_radius=6,
            fg_color="#0A0A0A",           # Black Background
            text_color="#FFFFFF",         # White Text
            border_color="#FFFFFF",       # White Border
            border_width=2,
            hover_color="#1A1A1A",        # Slight gray hover
            command=self.on_toggle
        )
        self.toggle_btn.pack(pady=(0, 20))

    def check_queue(self):
        """
        Polls the queue every 50ms without blocking the main GUI loop.
        """
        try:
            while True:  # Drain all items in the queue
                gesture, confidence = self.data_queue.get_nowait()
                self.gesture_label.configure(text=gesture.upper())
        except queue.Empty:
            pass
        finally:
            # Re-run this function every 50 milliseconds
            self.after(50, self.check_queue)

    def on_toggle(self):
        self.os_active = not self.os_active
        
        if self.os_active:
            # Swap colors: White button, Black text
            self.toggle_btn.configure(
                text="OS ACTIONS: ACTIVE",
                fg_color="#FFFFFF",
                text_color="#0A0A0A",
                hover_color="#E0E0E0"
            )
            self.worker.toggle_os_control(True)
        else:
            # Revert colors: Black button, White text
            self.toggle_btn.configure(
                text="ENABLE OS ACTIONS",
                fg_color="#0A0A0A",
                text_color="#FFFFFF",
                hover_color="#1A1A1A"
            )
            self.worker.toggle_os_control(False)


if __name__ == "__main__":
    app = GestureGUI()
    app.mainloop()