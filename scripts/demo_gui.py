import threading
import customtkinter as ctk
import time
import json
import os
import sys

# --- Pathing setup to ensure we can find the config, weights, and modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- Import Actual Pipeline ---
from inference_engine import InferenceEngine 
from utility.helper import find_setting_in_directory

class ModernGestureGUI(ctk.CTk):
    def __init__(self, inference_engine):
        super().__init__()
        
        self.engine = inference_engine
        
        # --- Window Setup ---
        self.title("Justisio Control Panel")
        self.geometry("450x550")
        self.configure(fg_color="#FFFFFF")  # Pure white background
        self.resizable(False, False)
        
        # --- Fonts ---
        self.title_font = ctk.CTkFont(family="Plus Jakarta Sans", size=26, weight="bold")
        self.heading_font = ctk.CTkFont(family="Plus Jakarta Sans", size=16, weight="bold")
        self.body_font = ctk.CTkFont(family="Plus Jakarta Sans", size=13)
        self.btn_font = ctk.CTkFont(family="Plus Jakarta Sans", size=14, weight="bold")

        # --- UI Construction ---
        self.build_ui()
        
    def build_ui(self):
        # 1. Header
        header_label = ctk.CTkLabel(
            self, 
            text="Selamat Datang\ndi Radar Control", 
            font=self.title_font, 
            text_color="#000000",
            justify="center"
        )
        header_label.pack(pady=(40, 30))

        # 2. Main Control Card (Mimicking the reference borders)
        self.card_frame = ctk.CTkFrame(
            self, 
            fg_color="transparent", 
            border_color="#000000", 
            border_width=2, 
            corner_radius=0
        )
        self.card_frame.pack(fill="x", padx=30, pady=(0, 15))

        self.status_title = ctk.CTkLabel(
            self.card_frame, 
            text="Status Stream: NONAKTIF", 
            font=self.heading_font, 
            text_color="#000000",
            anchor="w"
        )
        self.status_title.pack(fill="x", padx=15, pady=(15, 5))

        self.status_desc = ctk.CTkLabel(
            self.card_frame, 
            text="Radar sedang menguras buffer secara diam-diam. Tindakan OS dan inferensi PyTorch dijeda.", 
            font=self.body_font, 
            text_color="#555555",
            anchor="w",
            wraplength=350,
            justify="left"
        )
        self.status_desc.pack(fill="x", padx=15, pady=(0, 15))

        # 3. Settings / Options (Checkbox)
        self.debug_checkbox = ctk.CTkCheckBox(
            self, 
            text="Tampilkan log terminal", 
            font=self.body_font,
            text_color="#000000",
            fg_color="#000000",
            border_color="#000000",
            hover_color="#333333",
            corner_radius=4,
            command=self.toggle_debug # Wire checkbox to the engine's debug mode
        )
        self.debug_checkbox.pack(anchor="w", padx=30, pady=20)
        
        # Set initial UI state based on engine default
        if self.engine.debug_mode:
            self.debug_checkbox.select()

        # 4. Bottom Action Button (Solid Black)
        self.toggle_btn = ctk.CTkButton(
            self, 
            text="Mulai Deteksi", 
            font=self.btn_font,
            fg_color="#000000", 
            hover_color="#333333",
            text_color="#FFFFFF",
            corner_radius=0,
            height=45,
            command=self.toggle_stream
        )
        self.toggle_btn.pack(side="bottom", anchor="e", padx=30, pady=30)

    def toggle_debug(self):
        """Updates the engine's debug print state based on checkbox."""
        # .get() returns 1 if checked, 0 if unchecked. Convert to boolean.
        self.engine.debug_mode = bool(self.debug_checkbox.get())
        if self.engine.debug_mode:
            print("Log terminal DIAKTIFKAN.")
        else:
            print("Log terminal DINONAKTIFKAN.")

    def toggle_stream(self):
        """Toggles the engine's processing state and updates the UI."""
        if not self.engine.is_processing_active.is_set():
            # Turn ON
            self.engine.is_processing_active.set()
            self.toggle_btn.configure(text="Hentikan Deteksi")
            self.status_title.configure(text="Status Stream: AKTIF", text_color="#2E7D32") # Subtle green
            self.status_desc.configure(text="Inferensi PyTorch berjalan. Gestur akan dikirim ke OS Anda.")
            self.card_frame.configure(border_color="#2E7D32")
        else:
            # Turn OFF
            self.engine.is_processing_active.clear()
            self.toggle_btn.configure(text="Mulai Deteksi")
            self.status_title.configure(text="Status Stream: NONAKTIF", text_color="#000000")
            self.status_desc.configure(text="Radar sedang menguras buffer secara diam-diam. Tindakan OS dan inferensi PyTorch dijeda.")
            self.card_frame.configure(border_color="#000000")

    def on_closing(self):
        """Handles graceful shutdown of the background thread."""
        print("🛑 Menutup aplikasi...")
        self.engine.shutdown_flag.set()
        self.destroy()

# --- Bootloader ---
def launch_app():
    PORT = 9575
    
    # 1. Setup paths (Mirroring the backend's main block)
    cfg_path = os.path.join(root_dir, "radar_config", "config_3rx_2m")
    model_path = os.path.join(root_dir, "weights", "best_fmcw_model_v51_b32.pth")
    
    # Load settings
    setting_fn = find_setting_in_directory(cfg_path)
    with open(setting_fn, 'r') as f:
        setting = json.load(f)

    # 2. Initialize REAL Engine
    print("Memuat model dan inisialisasi engine...")
    engine = InferenceEngine(PORT, setting, model_path)
    
    # Ensure it starts paused (GUI controls it)
    engine.is_processing_active.clear()

    # 3. Start the Inference Engine in a background thread
    inference_thread = threading.Thread(target=engine.run, daemon=True)
    inference_thread.start()

    # 4. Start the GUI in the main thread
    app = ModernGestureGUI(engine)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    launch_app()