import tkinter as tk
import socket
import threading
from PIL import Image, ImageTk
import os

class LightweightDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("mmWave Gesture Demo")
        self.root.geometry("400x450")
        self.root.configure(bg="#2c3e50") # Clean, dark background
        
        # Keeps the window on top of other applications for the demo
        self.root.attributes("-topmost", True)

        # 1. Text Label
        self.gesture_label = tk.Label(root, text="Waiting for radar...", 
                                      font=("Segoe UI", 24, "bold"), 
                                      bg="#2c3e50", fg="#ecf0f1")
        self.gesture_label.pack(pady=20)

        # 2. Image Label
        self.image_label = tk.Label(root, bg="#2c3e50")
        self.image_label.pack(expand=True)

        # 3. Load Images into Memory
        self.image_cache = {}
        self.load_images()

        # 4. Background UDP Listener (Port 9576)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 9576))
        
        self.listen_thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.listen_thread.start()

        self.clear_timer = None

    def load_images(self):
        """Loads and caches images to prevent UI stutter during the demo."""
        gestures = ["Hand Away", "Hand Towards", "Swipe Down", "Swipe Left", "Swipe Right", "Swipe Up", "Idle"]
        
        # Ensure the assets directory exists
        os.makedirs("assets", exist_ok=True)
        
        for g in gestures:
            # Expects files like: assets/swipe_right.png
            filename = f"assets/{g.replace(' ', '_').lower()}.png"
            try:
                img = Image.open(filename).convert("RGBA")
                img = img.resize((250, 250), Image.Resampling.LANCZOS)
                self.image_cache[g] = ImageTk.PhotoImage(img)
            except FileNotFoundError:
                # Fallback: Create a blank grey square if the image is missing
                blank = Image.new('RGB', (250, 250), color='#34495e')
                self.image_cache[g] = ImageTk.PhotoImage(blank)

        self.update_display("Idle")

    def udp_listener(self):
        """Listens for the gesture string from the Inference Engine."""
        while True:
            data, _ = self.sock.recvfrom(1024)
            gesture = data.decode('utf-8')
            # Safely schedule the UI update on the main thread
            self.root.after(0, self.update_display, gesture)

    def update_display(self, gesture):
        """Updates the text and image on screen."""
        if gesture in self.image_cache:
            self.gesture_label.config(text=gesture)
            self.image_label.config(image=self.image_cache[gesture])
            
            # Automatically revert to "Idle" after 1.5 seconds
            if self.clear_timer is not None:
                self.root.after_cancel(self.clear_timer)
            
            if gesture != "Idle":
                self.clear_timer = self.root.after(1500, self.update_display, "Idle")

if __name__ == "__main__":
    root = tk.Tk()
    app = LightweightDemoGUI(root)
    root.mainloop()