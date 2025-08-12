import rospy
import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
def main():
    rospy.init_node('zed_yolo_node')
    # ROS publishers
    image_pub = rospy.Publisher('/zed/yolo/annotated_image', Image, queue_size=1)
    depth_pub = rospy.Publisher('/zed/yolo/detected_depths', Float32MultiArray, queue_size=1)
    bridge = CvBridge()
    # Load your YOLO model weights
    try:
        model = YOLO("best.pt")
    except Exception as e:
        rospy.logerr(f"Failed to load YOLO model: {e}")
        return
    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        rospy.logerr("Cannot open ZED camera")
        return
    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve image and depth data
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            # Get image data
            frame = image_zed.get_data()
            # Debug: Print original frame shape
            rospy.loginfo(f"Original frame shape: {frame.shape}")
            
            # Handle different channel formats from ZED
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    # Convert RGBA to RGB
                    frame = frame[:, :, :3]  # Simply drop alpha channel
                    rospy.loginfo("Converted RGBA to RGB by dropping alpha channel")
                elif frame.shape[2] == 3:
                    # Already 3 channels, assume it's BGR and convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rospy.loginfo("Converted BGR to RGB")
            
            # Debug: Print processed frame shape
            rospy.loginfo(f"Processed frame shape: {frame.shape}")
            
            # Get image dimensions for bounds checking
            height, width = frame.shape[:2]
            
            # Run YOLO detection (frame is now RGB format)
            results = model(frame, conf=0.5)
            detected_depths = []

            for result in results:
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Calculate center point of bounding box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    # Ensure coordinates are within image bounds
                    cx = max(0, min(cx, width - 1))
                    cy = max(0, min(cy, height - 1))

                    # Get depth value at center point
                    try:
                        depth_result = depth_zed.get_value(cx, cy)
                        
                        # ZED get_value typically returns a single float value for depth
                        if isinstance(depth_result, (tuple, list)):
                            # If it's a tuple/list, take the first element (depth)
                            depth_val = float(depth_result[0])
                        else:
                            # If it's a single value, use it directly
                            depth_val = float(depth_result)
                        
                        rospy.loginfo(f"Depth at ({cx}, {cy}): {depth_val}")
                        
                    except Exception as e:
                        rospy.logwarn(f"Error getting depth value: {e}")
                        depth_val = -1.0
                    
                    # Validate depth value
                    if depth_val > 0 and not np.isnan(depth_val) and not np.isinf(depth_val):
                        detected_depths.append(float(depth_val))
                        depth_str = f"{depth_val:.2f}m"
                    else:
                        detected_depths.append(-1.0)  # Invalid depth marker
                        depth_str = "N/A"
                    # Create label with class name, confidence, and depth
                    class_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                    label = f"{class_name} {conf:.2f} {depth_str}"
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label with background for better visibility
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    # Convert back to RGB for ROS publishing
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Convert to ROS Image message and publish
            try:
                ros_image = bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
                image_pub.publish(ros_image)
            except Exception as e:
                rospy.logwarn(f"Failed to publish image: {e}")
            # Publish detected depths as Float32MultiArray
            try:
                depth_msg = Float32MultiArray()
                depth_msg.data = detected_depths
                depth_pub.publish(depth_msg)
            except Exception as e:
                rospy.logwarn(f"Failed to publish depths: {e}")
        rate.sleep()
    # Clean up
    zed.close()
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ZED YOLO node interrupted")
    except Exception as e:
        rospy.logerr(f"Unexpected error in ZED YOLO node: {e}")
