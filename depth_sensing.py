""" Calculates the 3D distance from the camera's internal optical center to whatever object is 
visible at the center of the camera's field of view in milimeters 

ex: Distance to Camera at {640;360}: 1893.0105248513803
"""

import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    
    # i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Convert ZED image to OpenCV format
            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLOv8 object detection
            results = model(frame)[0]  # Get first frame result

            for obj in results.boxes.data:
                x1, y1, x2, y2, conf, cls = obj.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = model.names[int(cls)]  # Get object class

                # Calculate center of bounding box
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Get depth information at object center
                err, point_cloud_value = point_cloud.get_value(x_center, y_center)
                if err == sl.ERROR_CODE.SUCCESS and math.isfinite(point_cloud_value[2]):
                    distance = math.sqrt(sum(v ** 2 for v in point_cloud_value[:3]))
                    print(f"{class_name} detected at ({x_center},{y_center}), Distance: {distance:.2f} mm")

                    # Draw bounding box and distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name}: {distance:.2f} mm",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display image
            cv2.imshow("YOLOv8 Object Detection with ZED", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
