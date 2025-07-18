from pose_receiver import PoseReceiver
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Start PoseReceiver on the default host (0.0.0.0) and port (6001)
receiver = PoseReceiver(host='0.0.0.0', port=6001)
receiver.start()
