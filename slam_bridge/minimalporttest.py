import socket
HOST = "127.0.0.1"
PORT = 6001
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print("Listening...")
    conn, addr = s.accept()
    print("Connected by", addr)
    conn.close()
    