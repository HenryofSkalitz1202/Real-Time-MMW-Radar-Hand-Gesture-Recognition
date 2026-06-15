import json
import logging
import socket
import sys

from utility.BGT60TR13C import BGT60TR13C
from utility.helper import find_register_config_in_directory, find_setting_in_directory, calculate_frame_size

# Active info logs to monitor stream health
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main(IP_addr, port, cfg_dir, filename=None):
    logging.info(f"?? Starting strictly-aligned UDP Stream to {IP_addr}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    bgt60tr13c = BGT60TR13C(spi_speed=50_000_000, save_to_file=filename)
    bgt60tr13c.check_chip_id()
    
    reg_fn = find_register_config_in_directory(cfg_dir)
    bgt60tr13c.set_register_config_file(reg_fn)
    
    setting_fn = find_setting_in_directory(cfg_dir)
    with open(setting_fn, 'r') as file:
        setting_data = json.load(file)
        
    # calculate_frame_size returns total samples (e.g., 24576)
    frame_samples = calculate_frame_size(setting_data)
    
    # THE MATH FIX: Calculate exact network byte payload 
    # 1.5 bytes per 12-bit sample + 12-byte sequence header
    expected_bytes = int(frame_samples * 1.5) + 12 

    # FIFO setup relies on sample count, not byte count
    bgt60tr13c.set_fifo_parameters(frame_samples, 4096, 2048)
    bgt60tr13c.start()
    
    try:
        while True:
            # 1. PURGE THE CACHE (Zero-Lag Fix)
            # If the Pi reads SPI data faster than the network can send it,
            # drop the stale frames to maintain a true real-time feed.
            q_size = bgt60tr13c.frame_buffer.qsize()
            if q_size > 3:
                logging.warning(f"?? Backlog detected ({q_size} frames). Purging to maintain real-time sync!")
                while not bgt60tr13c.frame_buffer.empty():
                    sample_bytes = bgt60tr13c.frame_buffer.get()
            else:
                sample_bytes = bgt60tr13c.frame_buffer.get()

            # 2. STRICT ALIGNMENT VALIDATION (Garbage Float Fix)
            # Blocks truncated or fragmented packets from crashing the laptop's unpacker
            if len(sample_bytes) != expected_bytes:
                logging.error(f"? SPI Desync! Expected {expected_bytes} bytes, but got {len(sample_bytes)}. Dropping corrupted frame.")
                continue

            # 3. TRANSMIT
            sock.sendto(bytes(sample_bytes), (IP_addr, port))
            
    except KeyboardInterrupt:
        logging.info("?? Stopped by user")
        bgt60tr13c.stop()
    finally:
        sock.close()

if __name__ == "__main__":
    # Input the Laptop IP for the main function
    main("192.168.1.7", 9575, "radar_config/config_3rx_2m")
