import socket
import struct
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HOST = '0.0.0.0'
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"Listening on {HOST}:{PORT}...")

conn, addr = sock.accept()
print(f"Connected by {addr}")

x_vals, y_vals, z_vals = [], [], []

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def update(frame):
    data = recvall(conn, 48)
    if data is None:
        print("Connection closed")
        plt.close()
        return
    pose = struct.unpack('<12f', data)
    matrix = [pose[i*4:(i+1)*4] for i in range(3)]
    x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]
    print(f"Received translation: x={x}, y={y}, z={z}")
    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)
    plt.cla()
    plt.plot(x_vals, label='x')
    plt.plot(y_vals, label='y')
    plt.plot(z_vals, label='z')
    plt.xlabel('Frame')
    plt.ylabel('Translation (units)')
    plt.legend()
    plt.tight_layout()

fig = plt.figure()
ani = FuncAnimation(fig, update, interval=50)
plt.show()

conn.close()
sock.close()
