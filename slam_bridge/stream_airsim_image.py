import subprocess
import airsim
import socket
import struct
import numpy as np
import time
import os
import traceback
import sys
import logging
from pathlib import Path

# # Test log messages (add these for testing purposes)
# logging.info("Logging setup complete.")
# logging.debug("This is a debug message.")
# logging.error("This is an error message.")

PRINT_DEBUG_FRAMES = 10  # Only print debug info to console for first N frames

frame_count = 0

# print(f"[DEBUG] Current Working Directory: {os.getcwd()}", flush=True)

banner = "="*60 + "\n" + \
         "   AirSim → SLAM TCP Image Streamer (stream_airsim_image.py)   \n" + \
         f"   Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + \
         "="*60
logging.info(banner) 


VERBOSE = True

os.environ["OMP_NUM_THREADS"] = "2"

# Ensure flags/ and logs/ directories exist
Path("flags").mkdir(exist_ok=True)

def log(msg, level="info"):
    """
    Logs a message to both the console (for the first few frames if VERBOSE is True)
    and to the log file at the specified logging level.

    Args:
        msg (str): The message to log.
        level (str): The logging level ('debug', 'info', 'warning', 'error').
    """
    global frame_count
    if VERBOSE and frame_count < PRINT_DEBUG_FRAMES:
        print(msg, flush=True)

    if level == "debug":
        logging.debug(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    else:
        logging.info(msg)

def wait_for_slam_ready(flag_path="slam_ready.flag", timeout=15):
    log("[INFO] Waiting for SLAM to signal readiness...", level="info")
    logging.info("Waiting for SLAM to signal readiness...")
    start_time = time.time()
    while not os.path.exists(flag_path):
        if time.time() - start_time > timeout:
            print(f"[DEBUG] wait_for_slam_ready: Timeout after {timeout}s", flush=True)
            logging.error(f"Timeout waiting for {flag_path}")
            raise TimeoutError(f"[ERROR] Timeout waiting for {flag_path}")
        time.sleep(0.5)
    log("[INFO] slam_ready.flag found. Proceeding...", level="info")
    logging.info("slam_ready.flag found. Proceeding...")

def connect_with_retry(host, port, retries=10, delay=1):
    for i in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print("[DEBUG] Python streamer: TCP socket connect() succeeded", flush=True)
            log(f"[INFO] Connected to SLAM server on port {port}")
            return sock
        except socket.error as e:
            print(f"[DEBUG] connect_with_retry: Attempt {i+1} failed: {e}", flush=True)
            log(f"connect_with_retry: Attempt {i+1} failed: {e}", level="warning")
            # print(f"[WARN] Retry {i+1}/{retries} — Connection failed: {e}", flush=True)
            time.sleep(delay)
    print(f"[DEBUG] connect_with_retry: All {retries} attempts failed", flush=True)
    logging.error(f"Could not connect to {host}:{port} after {retries} attempts.")
    raise ConnectionRefusedError(f"[ERROR] Could not connect to {host}:{port} after {retries} attempts.")

def send_all(sock, data): # Ensure all data is sent
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            log("[DEBUG] send_all: Socket connection broken", level="debug")
            logging.error("send_all: Socket connection broken")
            raise RuntimeError("Socket connection broken")
        total_sent += sent

def get_wsl_ip():
    result = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True)
    return result.stdout.strip().split()[0]  # First IP address

def main():
    global frame_count
    # log("[DEBUG] Entering wait_for_slam_ready()")
    # try:
    #     wait_for_slam_ready()
    # except Exception as e:
    #     print(f"[DEBUG] Exception in wait_for_slam_ready: {e}", flush=True)
    #     logging.exception("Exception in wait_for_slam_ready")
    #     traceback.print_exc()
    #     sys.exit(1)  # crash visibly if no server
        
    log("[INFO] Waiting briefly before connecting to SLAM...")
    time.sleep(2)

    try:
        log("[DEBUG] Entering connect_with_retry()")
        # ip = get_wsl_ip()
        # sock = connect_with_retry(ip, 6000)
        sock = connect_with_retry("172.23.31.187", 6000)
        print("[INFO] Connected to SLAM server — sending first image...", flush=True)
    except Exception as e:
        print(f"[DEBUG] Exception in connect_with_retry: {e}", flush=True)
        logging.exception("Exception in connect_with_retry")
        traceback.print_exc()
        sys.exit(1)  # <-- Crash visibly

    log("[INFO] Connected to SLAM server — waiting for SLAM to finish initializing...")
    time.sleep(5) # Give SLAM time to initialize

    try:
        log("[DEBUG] Creating AirSim client")
        client = airsim.MultirotorClient()
        client.confirmConnection()
    except Exception as e:
        print(f"[DEBUG] Exception creating AirSim client: {e}", flush=True)
        logging.exception("Exception creating AirSim client")
        traceback.print_exc()
        sys.exit(1)  # <-- Crash visibly

    start_time = time.time()
    first_frame_sent = False

    try:
        time.sleep(0.5)
        log("[INFO] Waiting for valid image from AirSim...")
        for i in range(10):
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False)
                ])
            except Exception as e:
                print(f"[DEBUG] Exception in simGetImages (Scene): {e}", flush=True)
                logging.exception("Exception in simGetImages (Scene)")
                traceback.print_exc()
                responses = []
            if responses and responses[0].height > 0:
                log("[INFO] Valid image received from AirSim.")
                break
            print(f"[WARN] No valid image yet (attempt {i+1}/10)...", flush=True)
            log(f"No valid image yet (attempt {i+1}/10)...", level="warning")
            time.sleep(1)
        else:
            log("[ERROR] No valid image received after 10 retries. Exiting.", level="error")
            logging.error("No valid image received after 10 retries. Exiting.")
            sys.exit(1)  # <-- Crash visibly

        # --- Try sending the first image before entering the main loop ---
        try:
            log("[DEBUG] Sending first image to SLAM (pre-loop)")
            # TEMP: Add a delay before first image send for debugging startup timing
            time.sleep(2)
            # Touch a crash flag before attempting the first send
            with open("flags/streamer_crash.flag", "w") as f:
                f.write("stream_airsim_image.py crashed or was killed before first send\n")

            responses = client.simGetImages([
                airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("oakd_camera", airsim.ImageType.DepthPerspective, True)
            ])
            if frame_count < 3:
                print(f"[DEBUG] simGetImages returned: {[ (r.height, r.width) for r in responses ]}", flush=True)
            log(f"simGetImages returned: {[ (r.height, r.width) for r in responses ]}", level="debug")
            if not responses or responses[0].height == 0 or responses[1].height == 0:
                if frame_count < 3:
                    log("[WARN] Received empty image frame on first send, aborting...", level="warning")
                log("Received empty image frame on first send, aborting...", level="warning")
                sys.exit(1)  # <-- Crash visibly

            img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb.reshape(responses[0].height, responses[0].width, 3)
            img_depth = np.array(responses[1].image_data_float, dtype=np.float32)
            img_depth = img_depth.reshape(responses[1].height, responses[1].width)

            expected_rgb_bytes = responses[0].height * responses[0].width * 3
            expected_depth_bytes = responses[1].height * responses[1].width * 4

            if img_rgb.nbytes != expected_rgb_bytes:
                if frame_count < 3:
                    print(f"[DEBUG] RGB byte mismatch (first send): got {img_rgb.nbytes}, expected {expected_rgb_bytes}", flush=True)
                    print(f"[ERROR] RGB byte mismatch (first send): got {img_rgb.nbytes}, expected {expected_rgb_bytes}", flush=True)
                logging.error(f"RGB byte mismatch (first send): got {img_rgb.nbytes}, expected {expected_rgb_bytes}")
                sys.exit(1)  # <-- Crash visibly

            if img_depth.nbytes != expected_depth_bytes:
                if frame_count < 3:
                    print(f"[DEBUG] Depth byte mismatch (first send): got {img_depth.nbytes}, expected {expected_depth_bytes}", flush=True)
                    print(f"[ERROR] Depth byte mismatch (first send): got {img_depth.nbytes}, expected {expected_depth_bytes}", flush=True)
                logging.error(f"Depth byte mismatch (first send): got {img_depth.nbytes}, expected {expected_depth_bytes}")
                sys.exit(1)  # <-- Crash visibly

            # Try sending the first RGB and depth images
            try:
                # print("[DEBUG] Packing RGB header", flush=True)
                header = struct.pack('!III', responses[0].height, responses[0].width, img_rgb.nbytes)

                # print("[DEBUG] Sending RGB header", flush=True)
                send_all(sock, header)

                # print("[DEBUG] Sending RGB image bytes", flush=True)
                send_all(sock, img_rgb.tobytes())

                # print("[DEBUG] Packing depth header", flush=True)
                header = struct.pack('!III', responses[1].height, responses[1].width, img_depth.nbytes)

                # print("[DEBUG] Sending depth header", flush=True)
                send_all(sock, header)

                # print("[DEBUG] Sending depth image bytes", flush=True)
                try:
                    # print("[DEBUG] Sending depth image bytes", flush=True)
                    send_all(sock, img_depth.tobytes())
                    # print("[DEBUG] Depth image bytes sent successfully", flush=True)
                except Exception as e:
                    print(f"[ERROR] Exception sending depth image: {e}", flush=True)
                    traceback.print_exc()
                    sys.exit(1)

                # print("[DEBUG] About to create slam_ready.flag (pre-check passed)", flush=True)
                with open("flags/slam_ready.flag", "w") as f:
                    f.write("SLAM is ready\n")
                print("[DEBUG] slam_ready.flag created!", flush=True)

                try:
                    os.remove("flags/streamer_crash.flag")
                except Exception as cleanup_err:
                    print(f"[DEBUG] Could not remove streamer_crash.flag: {cleanup_err}", flush=True)

                first_frame_sent = True

            except Exception as e:
                print(f"[EXCEPTION] During first image send: {e}", flush=True)
                traceback.print_exc()
                sys.exit(1)


        except Exception as e:
            if frame_count < 3:
                print(f"[DEBUG] Exception preparing/sending first image: {e}", flush=True)
            logging.exception("Exception preparing/sending first image")
            traceback.print_exc()
            if frame_count < 3:
                print(f"[ERROR] Could not prepare/send first image to SLAM. Aborting.", flush=True)
            # Write a crash log file for debugging
            try:
                with open("flags/streamer_crash.log", "w") as f:
                    f.write(f"Exception during first image send:\n{e}\n")
                    import traceback as tb
                    tb.print_exc(file=f)
            except Exception as log_exc:
                if frame_count < 3:
                    print(f"[DEBUG] Failed to write streamer_crash.log: {log_exc}", flush=True)
                logging.error(f"Failed to write streamer_crash.log: {log_exc}")
            sys.exit(1)  # <-- Crash visibly

        # --- Main streaming loop ---
        while True:
            log("[DEBUG] Capturing images from AirSim...")
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.DepthPerspective, True)
                ])
            except Exception as e:
                if frame_count < 3:
                    print(f"[DEBUG] Exception in simGetImages (Scene+Depth): {e}", flush=True)
                logging.exception("Exception in simGetImages (Scene+Depth)")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            if not responses or responses[0].height == 0 or responses[1].height == 0:
                if frame_count < 3:
                    log("[WARN] Received empty image frame, skipping...", level="warning")
                log("Received empty image frame, skipping...", level="warning")
                time.sleep(0.1)
                continue

            try:
                img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_rgb.reshape(responses[0].height, responses[0].width, 3)

                img_depth = np.array(responses[1].image_data_float, dtype=np.float32)
                img_depth = img_depth.reshape(responses[1].height, responses[1].width)
                if frame_count < 3:
                    log(f"[DEBUG] AirSim image request successful: RGB shape {img_rgb.shape}, Depth shape {img_depth.shape}")
            except Exception as e:
                if frame_count < 3:
                    print(f"[DEBUG] Exception parsing image data: {e}", flush=True)
                logging.exception("Exception parsing image data")
                traceback.print_exc()
                if frame_count < 3:
                    print(f"[ERROR] Failed to parse image data: {e}", flush=True)
                continue

            expected_rgb_bytes = responses[0].height * responses[0].width * 3
            expected_depth_bytes = responses[1].height * responses[1].width * 4

            if img_rgb.nbytes != expected_rgb_bytes:
                if frame_count < 3:
                    print(f"[DEBUG] RGB byte mismatch: got {img_rgb.nbytes}, expected {expected_rgb_bytes}", flush=True)
                    print(f"[ERROR] RGB byte mismatch: got {img_rgb.nbytes}, expected {expected_rgb_bytes}", flush=True)
                logging.error(f"RGB byte mismatch: got {img_rgb.nbytes}, expected {expected_rgb_bytes}")
                sys.exit(1)  # <-- Crash visibly

            if img_depth.nbytes != expected_depth_bytes:
                if frame_count < 3:
                    print(f"[DEBUG] Depth byte mismatch: got {img_depth.nbytes}, expected {expected_depth_bytes}", flush=True)
                    print(f"[ERROR] Depth byte mismatch: got {img_depth.nbytes}, expected {expected_depth_bytes}", flush=True)
                logging.error(f"Depth byte mismatch: got {img_depth.nbytes}, expected {expected_depth_bytes}")
                sys.exit(1)  # <-- Crash visibly

            if frame_count < 3:
                log(f"[INFO] Sending images — RGB: {img_rgb.shape}, Depth: {img_depth.shape}")

            # Send RGB image
            try:
                header = struct.pack('!III', responses[0].height, responses[0].width, img_rgb.nbytes)
                send_all(sock, header)
                send_all(sock, img_rgb.tobytes())
                if not first_frame_sent and frame_count < 3:
                    log(f"[INFO] First image sent to SLAM at {time.time() - start_time:.2f} seconds")
                    first_frame_sent = True
                    
            except (BrokenPipeError, ConnectionAbortedError) as e:
                if frame_count < 3:
                    print(f"[DEBUG] Exception sending RGB: {e}", flush=True)
                logging.exception("Exception sending RGB")
                traceback.print_exc()
                if frame_count < 3:
                    print(f"[WARN] RGB send failed: {e}. Retrying once...", flush=True)
                time.sleep(1)
                try:
                    send_all(sock, img_rgb.tobytes())
                except Exception as e:
                    if frame_count < 3:
                        print(f"[DEBUG] RGB retry failed: {e}", flush=True)
                    logging.exception("RGB retry failed")
                    traceback.print_exc()
                    if frame_count < 3:
                        print(f"[ERROR] RGB retry failed: {e}", flush=True)
                    sys.exit(1)  # <-- Crash visibly

            # Send depth image
            try:
                header = struct.pack('!III', responses[1].height, responses[1].width, img_depth.nbytes)
                send_all(sock, header)
                send_all(sock, img_depth.tobytes())
            except (BrokenPipeError, ConnectionAbortedError) as e:
                if frame_count < 3:
                    print(f"[DEBUG] Exception sending depth: {e}", flush=True)
                logging.exception("Exception sending depth")
                traceback.print_exc()
                if frame_count < 3:
                    print(f"[WARN] Depth send failed: {e}", flush=True)
                sys.exit(1)  # <-- Crash visibly

            if frame_count < 3:
                log(f"Sent RGB ({img_rgb.shape}), Depth ({img_depth.shape})")
            frame_count += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        log("Stopping image streaming...")

    except Exception as e:
        print(f"[DEBUG] Unhandled exception in main loop: {e}", flush=True)
        logging.exception("Unhandled exception in main loop")
        traceback.print_exc()
        sys.exit(1)  # <-- Crash visibly

    finally:
        log("[DEBUG] Closing socket")
        sock.close()

if __name__ == "__main__":
    main()
