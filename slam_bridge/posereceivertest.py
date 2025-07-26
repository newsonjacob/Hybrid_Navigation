import socket
import struct

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 6001       # Match POSE_RECEIVER_PORT in your C++ code

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening for connection on {HOST}:{PORT}...")
    conn, addr = s.accept()
    print(f"Accepted connection from {addr}")

    while True:
        data = conn.recv(48)  # Expecting 12 floats (48 bytes)
        if not data:
            print("Connection closed.")
            break
        floats = struct.unpack('<12f', data)
        print("Received pose:", floats)
        