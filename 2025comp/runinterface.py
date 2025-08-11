import pyzed.sl as sl
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('2025comp/best.pt')  # path to your weights


# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # or HD1080
init_params.camera_fps = 30

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Cannot open ZED camera")
    exit(1)

image = sl.Mat()

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()  # NumPy array in RGB format

            # Run YOLO inference
            results = model.predict(frame)

            # Draw boxes on the frame
            annotated_frame = results[0].plot()

            # Convert RGB to BGR for OpenCV display
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('YOLO on ZED 2i', annotated_frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
