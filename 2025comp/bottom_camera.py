import rospy
import pyzed.sl as sl
import numpy as np
from std_msgs.msg import Float32MultiArray
# from geometry_msgs.msg import PointStamped

def main():
    # Check ROS environment before initializing
    try:
        import os
        ros_master_uri = os.environ.get('ROS_MASTER_URI', 'Not set')
        rospy.loginfo(f"ROS_MASTER_URI: {ros_master_uri}")
        
        rospy.init_node('zed_bottom_depth_node')
        rospy.loginfo("ROS node initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize ROS node: {e}")
        return
    
    # ROS publishers
    ground_depth_pub = rospy.Publisher('/zed/bottom/ground_depth', Float32MultiArray, queue_size=1)
    
    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.1  # Minimum depth in meters
    init_params.depth_maximum_distance = 10.0  # Maximum depth in meters
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        rospy.logerr("Cannot open ZED camera")
        return
    
    runtime_params = sl.RuntimeParameters()
    depth_zed = sl.Mat()
    
    rate = rospy.Rate(30)  # 30 Hz
    
    while not rospy.is_shutdown():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve depth data
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            
            # Get depth data as numpy array
            depth_data = depth_zed.get_data()
            
            # Get image dimensions
            height, width = depth_data.shape[:2]
            
            # Calculate center coordinates
            center_x = width // 2
            center_y = height // 2
            
            # Sample 3x3 area around center
            depth_values = []
            for dy in range(-1, 2):  # -1, 0, 1
                for dx in range(-1, 2):  # -1, 0, 1
                    try:
                        depth_result = depth_zed.get_value(center_x + dx, center_y + dy)
                        
                        depth_val = -1.0
                        if isinstance(depth_result, (tuple, list)):
                            error_code, depth_value = depth_result
                            if str(error_code) == 'SUCCESS':
                                try:
                                    depth_val = float(depth_value)
                                except (ValueError, TypeError):
                                    continue
                        else:
                            try:
                                depth_val = float(depth_result)
                            except (ValueError, TypeError):
                                continue
                        
                        # Only include valid depth values
                        if depth_val > 0 and not np.isnan(depth_val) and not np.isinf(depth_val):
                            depth_values.append(depth_val)
                    
                    except Exception as e:
                        continue
            
            # Calculate average ground depth
            if depth_values:
                avg_ground_depth = sum(depth_values) / len(depth_values)
                # min_ground_depth = min(depth_values)
                # max_ground_depth = max(depth_values)
                
                rospy.loginfo(f"Ground depth - Avg: {avg_ground_depth:.2f}m")
                
                # Publish ground depth data [avg_depth, min_depth, max_depth, num_valid_pixels]
                try:
                    ground_msg = Float32MultiArray()
                    ground_msg.data = float(avg_ground_depth)
                    ground_depth_pub.publish(ground_msg)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish ground depth: {e}")
            else:
                rospy.logwarn("No valid depth readings from center 3x3 area")
                
                # Publish invalid data
                try:
                    ground_msg = Float32MultiArray()
                    ground_msg.data = [-1.0, -1.0, -1.0, 0.0]  # Invalid depth markers
                    ground_depth_pub.publish(ground_msg)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish invalid ground depth: {e}")
        
        rate.sleep()
    
    # Clean up
    zed.close()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ZED bottom depth node interrupted")
    except Exception as e:
        rospy.logerr(f"Unexpected error in ZED bottom depth node: {e}")