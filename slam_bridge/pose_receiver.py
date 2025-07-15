import socket
import struct
import threading
import time
from collections import deque
from typing import Optional, Tuple, List

import logging

logger = logging.getLogger("pose_receiver")

class PoseReceiver:
    """TCP server that receives 3x4 pose matrices and stores the latest pose."""

    def __init__(self, host: str = "0.0.0.0", port: int = 6001, history_size: int = 500) -> None:
        self.host = host
        self.port = port
        self.history_size = history_size
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_pose: Optional[List[List[float]]] = None
        self._latest_inliers: Optional[int] = None
        self._latest_covariance: Optional[float] = None
        self._history = deque(maxlen=history_size)
        self._conn: Optional[socket.socket] = None


    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info(f"[PoseReceiver] Preparing to bind to {(self.host, self.port)}")
        self._sock.bind((self.host, self.port))
        logger.info(f"[PoseReceiver] Bound successfully on {(self.host, self.port)}")
        self.port = self._sock.getsockname()[1]
        self._sock.listen(1)
        self._sock.settimeout(1)

        logger.info(f"[PoseReceiver] Listening on {self.host}:{self.port}...")
    
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None: # Stop the receiver and clear resources.
        logger.info("[PoseReceiver] Stopping receiver...")
        self._stop_event.set()  # Signal the thread to stop.

        # Wake accept() by connecting then close the socket
        if self._sock:
            try:
                try:
                    socket.create_connection((self.host, self.port), timeout=0.1).close()
                except Exception:
                    pass
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        # Ensure any active connection is closed
        if self._conn:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

        # Wait for the thread to finish, warn if it doesn't exit
        if self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("[PoseReceiver] Thread did not exit cleanly after stop().")

    def get_latest_pose(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns the translation (x, y, z) from the latest received 3x4 pose matrix.

        Returns:
            Optional[Tuple[float, float, float]]: The translation components (x, y, z) if available, otherwise None.
        """
        with self._lock: # Ensure thread-safe access to the latest pose
            logger.info(f"[PoseReceiver] Latest pose raw: {self._latest_pose}")
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

    def get_latest_inliers(self) -> Optional[int]:
        with self._lock:
            return self._latest_inliers

    def get_latest_covariance(self) -> Optional[float]:
        with self._lock:
            return self._latest_covariance

    def get_pose_history(self):
        return list(self._history)

    # Receives a 3x4 pose matrix in binary format (12 floats, little-endian)
    def _recv_loop(self) -> None:
        logger.info("[PoseReceiver] Starting receive loop...")
        assert self._sock is not None  # Ensure the socket is initialized
        logger.info(f"[PoseReceiver] Listening on {self.host}:{self.port}")

        retry_count = 0
        max_retries = None
        
        logger.info(f"[PoseReceiver] stop_event is set? {self._stop_event.is_set()}")
        while not self._stop_event.is_set():
            try:
                # Set the socket to block until a connection is made
                self._sock.settimeout(None)  # block forever until first connection
                logger.info(f"[PoseReceiver] Waiting for connection (retry {retry_count})...")
                self._conn, addr = self._sock.accept()
                logger.info(f"[PoseReceiver] Accepted connection from {addr}")
                self._conn.settimeout(1)
                retry_count = 0

                # Start receiving data from the client
                while not self._stop_event.is_set():
                    data = self._recvall(self._conn, 48)
                    if not data:
                        logger.warning("[PoseReceiver] Client disconnected or sent no data.")
                        break

                    # Expecting 12 floats (3x4 matrix) = 48 bytes
                    pose = struct.unpack('<12f', data)
                    matrix = [list(pose[i * 4:(i + 1) * 4]) for i in range(3)]
                    with self._lock:
                        self._latest_pose = matrix
                        self._history.append((time.time(), matrix))

                    # --- Receive covariance (float, 4 bytes) ---
                    cov_data = self._recvall(self._conn, 4)
                    if cov_data and len(cov_data) == 4:
                        self._latest_covariance = struct.unpack('<f', cov_data)[0]
                        logger.debug(f"[PoseReceiver] Received covariance: {self._latest_covariance}")
                    else:
                        self._latest_covariance = None
                        logger.warning("[PoseReceiver] Covariance data missing or incomplete.")

                    # --- Receive inliers (int, 4 bytes) ---
                    inlier_data = self._recvall(self._conn, 4)
                    if inlier_data and len(inlier_data) == 4:
                        self._latest_inliers = struct.unpack('<i', inlier_data)[0]
                        logger.debug(f"[PoseReceiver] Received inlier count: {self._latest_inliers}")
                    else:
                        self._latest_inliers = None
                        logger.warning("[PoseReceiver] Inlier data missing or incomplete.")

                    # Log the received pose
                    tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]
                    print(f"[PoseReceiver] Received Twc translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
                    logger.debug(f"[PoseReceiver] Received Twc translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

                # Clean up connection after client disconnects
                if self._conn:
                    self._conn.close()
                    self._conn = None

            except Exception as e:
                retry_count += 1
                logger.error(f"[PoseReceiver] accept() or recv loop error (retry {retry_count}): {e}")
                if max_retries is not None and retry_count >= max_retries:
                    logger.error("[PoseReceiver] Maximum retries reached, stopping receiver loop.")
                    break
                time.sleep(2)

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

