# test_socket_connect_and_send.py
import socket
import struct
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("172.23.31.187", 6000))
print("[✅] Connected")

# Send dummy RGB header: 720 height, 1280 width, 2764800 bytes (720*1280*3)
rgb_header = struct.pack("!III", 720, 1280, 720*1280*3)
sock.send(rgb_header)

# Send dummy RGB data
sock.send(b'\x00' * (720*1280*3))

time.sleep(1)
sock.close()
