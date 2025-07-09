import socket
import struct
import time
import threading
from collections import deque
import numpy as np
import logging
from slam_bridge.logging_helper import configure_file_logger

logger = configure_file_logger("slam_receiver.log")

HOST = "192.168.1.102"  # Default IP if not provided
PORT = 6001

slam_pose = {
    'pose_matrix': None,
    'timestamp': None,
    'valid': False,
    'lock': threading.Lock()
}

pose_history = deque(maxlen=500)
frame_counter = 0

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def _recv_loop(host: str, port: int):
    logger.info("Starting...")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # Create a new socket
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.listen(1)
                sock.settimeout(5)

                try:
                    conn, addr = sock.accept()
                    logger.info("\u2705 Connected by %s", addr)
                    with conn:
                        global frame_counter
                        while True:
                            data = recvall(conn, 48) # 12 floats * 4 bytes each = 48 bytes
                            if data is None:
                                logger.info("Connection closed.")
                                break
                            pose = struct.unpack('<12f', data) # Unpack 12 floats (3x4 matrix)
                            matrix = np.array([pose[i*4:(i+1)*4] for i in range(3)], dtype=np.float32)
                            with slam_pose['lock']:
                                slam_pose['pose_matrix'] = matrix.tolist()
                                slam_pose['timestamp'] = time.time()
                                slam_pose['valid'] = True
                            is_identity = np.allclose(matrix, np.eye(3,4), atol=1e-6)
                            logger.info("Frame %d valid=%s", frame_counter, not is_identity)
                            logger.info("%s", matrix)
                            pose_history.append((frame_counter, slam_pose['timestamp'], matrix, not is_identity))
                            frame_counter += 1
                except socket.timeout:
                    pass
        except Exception as e:
            logger.error("Error: %s", e)

def start_receiver(host: str = HOST, port: int = PORT):
    threading.Thread(target=_recv_loop, args=(host, port), daemon=True).start()

def get_latest_pose():
    with slam_pose['lock']:
        if slam_pose['valid']:
            x = slam_pose['pose_matrix'][0][3]
            y = slam_pose['pose_matrix'][1][3]
            z = slam_pose['pose_matrix'][2][3]
            return (x, y, z)
        else:
            return None

def get_pose_history():
    return list(pose_history)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SLAM pose receiver")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    logger.info("Waiting for SLAM client...")
    start_receiver(args.host, args.port)
    while True:
        time.sleep(1)  # Keep the script alive

