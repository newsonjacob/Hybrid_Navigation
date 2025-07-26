import airsim
import numpy as np
import cv2

def test_camera(camera_names=["front_left", "front_right"], image_type=airsim.ImageType.Scene):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Requesting images from cameras: {camera_names}...")
    requests = [airsim.ImageRequest(name, image_type, False, True) for name in camera_names]
    responses = client.simGetImages(requests)
    if not responses or any(r is None or len(r.image_data_uint8) == 0 for r in responses):
        print("No image data received from one or more cameras.")
        return
    for idx, (name, response) in enumerate(zip(camera_names, responses)):
        print(f"{name} raw data length: {len(response.image_data_uint8)}")
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to decode image from camera {name}.")
            continue
        print(f"Camera {name} image shape: {img.shape}")
        cv2.imwrite(f"camera_{name}_scene.png", img)
        print(f"Saved image as camera_{name}_scene.png")

if __name__ == "__main__":
    # Try different camera names as needed
    test_camera(camera_names=["front_left", "front_right"])
    