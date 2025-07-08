import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect(("127.0.0.1", 6000))
    print("[✅] Connected to port 6000")
except Exception as e:
    print("[❌] Failed to connect:", e)
finally:
    sock.close()
