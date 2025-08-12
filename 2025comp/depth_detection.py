import rospy
import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R

def main():
    rospy.init_node('zed_yolo_node')
    
    # ROS publisher - only need coordinates now
    coords_pub = rospy.Publisher('/zed/detections', Float32MultiArray, queue_size=1)
    gyro_pub = rospy.Publisher('/zed/gyro', Float32MultiArray, queue_size=1)
    displacement_pub = rospy.Publisher('/zed/displacement', Float32MultiArray, queue_size=1)
    
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
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_as_static = False
    if zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
        rospy.logerr("Failed to enable positional tracking")
        zed.close()
        return
    pose_zed = sl.Pose()
    
    rate = rospy.Rate(30)  # 30 Hz
    
    while not rospy.is_shutdown():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve image and depth data
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            
            
            # Get image data
            frame = image_zed.get_data()
            
            # Handle different channel formats from ZED
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    # Convert RGBA to RGB
                    frame = frame[:, :, :3]  # Simply drop alpha channel
                elif frame.shape[2] == 3:
                    # Already 3 channels, assume it's BGR and convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions for bounds checking
            height, width = frame.shape[:2]
            
            # Run YOLO detection (frame is now RGB format)
            results = model(frame, conf=0.6)
            detections = []  # Store [x1, x2, y1, y2, depth, class_id, confidence] for each detection

            for result in results:
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Normalize coordinates to 0-1 range
                    x1_norm = x1 / float(width - 1)
                    x2_norm = x2 / float(width - 1)
                    y1_norm = y1 / float(height - 1)
                    y2_norm = y2 / float(height - 1)
                    
                    # Calculate center point of bounding box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Ensure coordinates are within image bounds
                    cx = max(0, min(cx, width - 1))
                    cy = max(0, min(cy, height - 1))

                    # Get depth value at center point
                    try:
                        depth_result = depth_zed.get_value(cx, cy)
                        
                        # Handle different return formats and check for error codes
                        depth_val = -1.0  # Default to invalid
                        
                        if isinstance(depth_result, (tuple, list)):
                            # It's a tuple/list - ZED returns (error_code, depth_value)
                            error_code, depth_value = depth_result
                            if str(error_code) != 'SUCCESS':
                                rospy.logwarn(f"Depth measurement failed at ({cx}, {cy}): {error_code}")
                            else:
                                try:
                                    depth_val = float(depth_value)
                                except (ValueError, TypeError):
                                    rospy.logwarn(f"Could not convert depth value to float at ({cx}, {cy}): {depth_value}")
                        else:
                            # Single value, check if it's an error code
                            if hasattr(depth_result, 'name') and 'ERROR' in str(depth_result):
                                rospy.logwarn(f"Depth measurement failed at ({cx}, {cy}): {depth_result}")
                            else:
                                try:
                                    depth_val = float(depth_result)
                                except (ValueError, TypeError):
                                    rospy.logwarn(f"Could not convert depth value to float at ({cx}, {cy}): {depth_result}")
                        
                        # Log successful depth readings
                        if depth_val > 0 and not np.isnan(depth_val) and not np.isinf(depth_val):
                            rospy.loginfo(f"Detection: {model.names[cls]} at ({cx}, {cy}) depth: {depth_val:.2f}m, confidence: {conf:.3f}")
                        
                    except Exception as e:
                        import traceback
                        error_type = type(e).__name__
                        error_msg = str(e)
                        
                        rospy.logwarn(f"Unexpected error getting depth value at ({cx}, {cy}):")
                        rospy.logwarn(f"  Error Type: {error_type}")
                        rospy.logwarn(f"  Error Message: {error_msg}")
                        depth_val = -1.0
                    
                    # Validate depth value
                    if depth_val <= 0 or np.isnan(depth_val) or np.isinf(depth_val):
                        depth_val = -1.0  # Invalid depth marker
                    
                    # Store detection info: [x1, x2, y1, y2, depth, class_id, confidence]
                    detections.extend([float(x1_norm), float(x2_norm), float(y1_norm), float(y2_norm), 
                                     float(cls), float(depth_val),  float(conf)])

            # Publish detections as Float32MultiArray
            try:
                detection_msg = Float32MultiArray()
                detection_msg.data = detections
                coords_pub.publish(detection_msg)
            except Exception as e:
                rospy.logwarn(f"Failed to publish detections: {e}")

            if zed.get_position(pose_zed, sl.REFERENCE_FRAME.WORLD) == sl.POSITIONAL_TRACKING_STATE.OK:
                translation = pose_zed.get_translation().get()
                orientation = pose_zed.get_orientation().get()
                r = R.from_quat(orientation)
                euler_angles = r.as_euler('xyz', degrees=True)
                try:
                    gyro_msg = Float32MultiArray()
                    displacement_msg = Float32MultiArray()
                    gyro_msg.data = euler_angles
                    displacement_msg.data = translation
                    gyro_pub.publish(gyro_msg)
                    displacement_pub.publish(displacement_msg)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish IMU data: {e}")
        
        rate.sleep()
    
    # Clean up
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ZED YOLO node interrupted")
    except Exception as e:
        rospy.logerr(f"Unexpected error in ZED YOLO node: {e}")