import socket
import struct
import threading
import time
from collections import deque
from typing import Optional, Tuple, List

import logging

logger = logging.getLogger(__name__)

class PoseReceiver:
    """TCP server that receives 3x4 pose matrices and stores the latest pose."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6001, history_size: int = 500) -> None:
        self.host = host
        self.port = port
        self.history_size = history_size
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_pose: Optional[List[List[float]]] = None
        self._history = deque(maxlen=history_size)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(1)
        self._sock.settimeout(1)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                # Wake accept() by connecting then close the socket
                try:
                    socket.create_connection((self.host, self.port), timeout=0.1).close()
                except Exception:
                    pass
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=1)

    def get_latest_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            logger.debug(f"[PoseReceiver] Latest pose raw: {self._latest_pose}")
            if self._latest_pose is None:
                return None
            try:
                x = self._latest_pose[0][3]
                y = self._latest_pose[1][3]
                z = self._latest_pose[2][3]
                return (x, y, z)
            except Exception as e:
                logger.error(f"[PoseReceiver] Pose extraction failed: {e}")
                return None


    def get_pose_history(self):
        return list(self._history)

    # Receives a 3x4 pose matrix in binary format (12 floats, little-endian)
    def _recv_loop(self) -> None: 
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                try:
                    conn, _ = self._sock.accept()
                    logger.info("[PoseReceiver] Client connected.")
                except socket.timeout:
                    continue
                conn.settimeout(1) 
                with conn: 
                    while not self._stop_event.is_set():
                        data = self._recvall(conn, 48)
                        logger.debug(f"[PoseReceiver] Raw pose bytes received: {data}")
                        if not data:
                            break
                        pose = struct.unpack('<12f', data)
                        logger.debug(f"[PoseReceiver] Decoded pose: {pose}")
                        matrix = [list(pose[i*4:(i+1)*4]) for i in range(3)]
                        with self._lock:
                            self._latest_pose = matrix
                            self._history.append((time.time(), matrix))

                        # Log the received translation for debugging
                        tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]
                        logger.debug(f"[PoseReceiver] Received Tcw translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            except OSError:
                break
            except Exception as e:
                logger.error("Error: %s", e)
        logger.info("PoseReceiver stopped")

    def _recvall(self, conn: socket.socket, n: int) -> bytes:
        data = b''
        while len(data) < n and not self._stop_event.is_set():
            try:
                packet = conn.recv(n - len(data))
            except socket.timeout:
                continue
            if not packet:
                return b''
            data += packet
        return data