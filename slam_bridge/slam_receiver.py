import socket
import struct
import time
import threading

HOST = "192.168.1.102"  # Default IP if not provided
PORT = 6001

slam_pose = {
    'pose_matrix': None,
    'timestamp': None,
    'valid': False,
    'lock': threading.Lock()
}

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def _recv_loop(host: str, port: int):
    print("[SLAM Receiver] Starting...")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # Create a new socket
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.listen(1)
                sock.settimeout(5)

                try:
                    conn, addr = sock.accept()
                    print(f"[SLAM Receiver] ✅ Connected by {addr}")
                    with conn:
                        while True:
                            data = recvall(conn, 48) # 12 floats * 4 bytes each = 48 bytes
                            if data is None:
                                print("[SLAM Receiver] Connection closed.")
                                break
                            pose = struct.unpack('<12f', data) # Unpack 12 floats (3x4 matrix)
                            with slam_pose['lock']:
                                slam_pose['pose_matrix'] = [pose[i*4:(i+1)*4] for i in range(3)]
                                slam_pose['timestamp'] = time.time()
                                slam_pose['valid'] = True
                except socket.timeout:
                    pass
        except Exception as e:
            print(f"[SLAM Receiver] Error: {e}")

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SLAM pose receiver")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    print("[SLAM Receiver] Waiting for SLAM client...")
    start_receiver(args.host, args.port)
    while True:
        time.sleep(1)  # Keep the script alive

