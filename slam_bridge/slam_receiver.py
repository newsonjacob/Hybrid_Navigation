from typing import Optional, Tuple

from .pose_receiver import PoseReceiver
import logging
from uav.logging_config import setup_logging

logger = logging.getLogger(__name__)

HOST = "192.168.1.102"  # Default IP if not provided
PORT = 6001

_receiver: Optional[PoseReceiver] = None

def start_receiver(host: str = HOST, port: int = PORT) -> None:
    """Start the global PoseReceiver on the given host/port."""
    global _receiver
    if _receiver is None:
        _receiver = PoseReceiver(host, port)
        _receiver.start()

def stop_receiver() -> None:
    """Stop the global PoseReceiver if running."""
    global _receiver
    if _receiver is not None:
        _receiver.stop()
        _receiver = None

def get_latest_pose() -> Optional[Tuple[float, float, float]]:
    """Return the latest pose from the global receiver."""
    if _receiver is not None:
        return _receiver.get_latest_pose()
    return None


def get_pose_history():
    if _receiver is not None:
        return _receiver.get_pose_history()
    return []


if __name__ == "__main__":
    import argparse
    import time
    setup_logging(None)

    parser = argparse.ArgumentParser(description="SLAM pose receiver")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    
    start_receiver(args.host, args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_receiver()

