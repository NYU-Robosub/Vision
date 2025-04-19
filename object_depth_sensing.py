import pyzed.sl as sl
import cv2
import numpy as np
import math
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # or yolov8s.pt, etc.

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # Convert ZED image to OpenCV format
            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLOv8
            results = model(frame)[0]

            for obj in results.boxes.data:
                x1, y1, x2, y2, conf, cls = obj.cpu().numpy()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                class_name = model.names[int(cls)]

                # Extract depth values within the bounding box
                depth_array = depth.get_data()
                box_depths = []
                for y in range(y1, y2):
                    if 0 <= y < depth_array.shape[0]:
                        for x in range(x1, x2):
                            if 0 <= x < depth_array.shape[1]:
                                z_val = depth_array[y][x]
                                if math.isfinite(z_val) and z_val > 0:
                                    box_depths.append(z_val)

                # Print stats or array
                if box_depths:
                    print(f"{class_name} @ ({x1},{y1}) to ({x2},{y2}) → Median Depth: {np.median(box_depths):.2f} mm")
                    # To see raw values: print(box_depths)

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the image - delete later
            cv2.imshow("Object Detection with Depth", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
