import cv2
import csv
import numpy as np
import math
import sys
from typing import Optional, Tuple
from vmbpy import *
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, CameraCapture, VehicleAttitude, SensorGps
from geometry_msgs.msg import Twist
from goonvision_functions import *

last_x = None
last_y = None
last_z = None
last_roll = None
last_pitch = None
last_yaw = None
last_lat = None
last_lon = None

CSV_FILENAME = "lines_log.csv"
class DroneDataLogger(Node):
    def __init__(self):
        super().__init__('drone_data_logger')

        # # Open CSV file and write the header
        # with open(CSV_FILENAME, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Frame Timestamp", "X Position", "Y Position", "Z Position", 
        #                      "Roll (deg)", "Pitch (deg)", "Yaw (deg)"])

        # ROS2 Subscriptions
        #self.create_subscription(CameraCapture, '/fmu/out/camera_capture', self.camera_callback, 10)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.position_callback, 10)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, 10)
        self.create_subscription(SensorGps,'/fmu/out/vehicle_gps_position',self.gps_callback,10)
        # Data storage
        self.last_camera_timestamp = None
        self.last_x = None
        self.last_y = None
        self.last_z = None
        self.last_roll = None
        self.last_pitch = None
        self.last_yaw = None
        self.last_lat = None
        self.last_lon = None
        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    #def camera_callback(self, msg):
    #   """ Store camera frame timestamp """
    #    self.last_camera_timestamp = msg.timestamp
    #    self.save_data()

    def position_callback(self, msg):
        """ Store drone position """
        last_x = msg.x
        last_y = msg.y
        last_z = msg.z

    def attitude_callback(self, msg):
        """ Store drone orientation """
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        roll, pitch, yaw = self.quaternion_to_euler(q)
        last_roll = np.degrees(roll)
        last_pitch = np.degrees(pitch)
        last_yaw = np.degrees(yaw)

    def gps_callback(self, msg):
        """ Store camera frame timestamp """
        last_lat = msg.latitude_deg
        last_lon = msg.longitude_deg

    def quaternion_to_euler(self, q):
        """ Convert quaternion to Euler angles """
        w, x, y, z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    # def save_data(self):
    #     """ Save synchronized frame metadata to CSV """
    #     if self.last_camera_timestamp is not None and self.last_x is not None:
    #         with open(CSV_FILENAME, 'a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([self.last_camera_timestamp, 
    #                              self.last_x, self.last_y, self.last_z, 
    #                              self.last_roll, self.last_pitch, self.last_yaw])
    #         self.get_logger().info(f"Logged Frame {self.last_camera_timestamp}")

    def timer_callback(self) -> None:
        #print((self.last_roll,self.last_pitch,self.last_yaw))
        pass

# Constants
FOV_X = math.radians(65.5)  # Horizontal Field of View (radians)
FOV_Y = math.radians(51.4)  # Vertical Field of View (radians)
FRAME_WIDTH = 2592   # Camera frame width (pixels)
FRAME_HEIGHT = 1944  # Camera frame height (pixels)
altitude = 100.0    # altitude of the plane (meters), adjust as needed
lines = []
def calculate_position_and_angle(cx, cy):
    """Convert pixel coordinates to real-world distances and calculate angles."""
    # Convert pixel positions to real-world distances using FOV and altitude
    Dx = altitude * math.tan(FOV_X)  # Real-world width covered by the camera
    Dy = altitude * math.tan(FOV_Y)  # Real-world height covered by the camera

    # Map pixel coordinates to real-world coordinates (center is (0,0))
    dx = ((cx - FRAME_WIDTH / 2) / FRAME_WIDTH) * Dx
    dy = ((cy - FRAME_HEIGHT / 2) / FRAME_HEIGHT) * Dy

    # Position vector relative to the plane
    r = np.array([dx, dy, altitude])

    # Calculate angles (in degrees)
    thetaX = math.degrees(math.atan2(dx, altitude))  # Angle in x-direction (wingspan)
    thetaY = -math.degrees(math.atan2(dy, altitude))  # Angle in y-direction (length of plane)

    return r, thetaX, thetaY

def generate_line(cx, cy):
    # Compute angles based on FOV
    thetaX = FOV_X * ((cx - FRAME_WIDTH / 2) / FRAME_WIDTH)
    thetaY = -FOV_Y * ((cy - FRAME_WIDTH / 2) / FRAME_WIDTH)

    # Define position (anchor point)
    position = np.array([last_lon, last_lat,last_z])

    # Convert angles to direction vector
    direction = np.array([
        np.cos(last_yaw) * np.cos(thetaY + last_roll),
        np.sin(last_yaw) * np.cos(thetaY + last_roll),
        np.sin(thetaX + last_pitch)
    ])

    # Normalize the direction vector to ensure unit length
    direction /= np.linalg.norm(direction)

    # Append the line to CSV (x, y, z, dx, dy, dz)
    with open(CSV_FILENAME, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([*position, *direction])

    return position, direction


def process_frame(frame):
    """Detect white dot and calculate angles."""
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest detected contour (assuming only 1 bright dot)
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Compute real-world position and angles
            r, thetaX, thetaY = calculate_position_and_angle(cx, cy)

            # Display results
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            text = f"({cx},{cy}) AngleX: {thetaX:.2f}° AngleY: {thetaY:.2f}°"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lines.append(generate_line(cx,cy))
            print(f"Dot at {r}, AngleX: {thetaX:.2f}°, AngleY: {thetaY:.2f}°")

            #height, width = frame.shape[:2]
            #print(f"Frame Dimensions: Width = {width} px, Height = {height} px")
    # Display the frame
    #cv2.imshow('Dot Tracking & Angle Calculation', frame)
    cv2.waitKey(1)

def frame_handler(cam: Camera, stream: Stream, frame: Frame):
    """Process each frame acquired from the Allied Vision camera."""
    
    frame_data = frame.as_numpy_ndarray()

    if frame_data is not None:
        # Convert raw camera data to OpenCV format
        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_BAYER_BG2BGR)
        
        # Process the frame
        process_frame(frame_bgr)

    # Queue the frame again for next acquisition
    cam.queue_frame(frame)

def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                print(f"Failed to access Camera '{camera_id}'. Abort.")
                sys.exit(1)
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                print("No Cameras accessible. Abort.")
                sys.exit(1)
            return cams[0]

def setup_camera(cam: Camera):
    """Configures the camera before starting acquisition."""
    
    with cam:
        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VmbFeatureError):
            pass

def main(args=None) -> None:
    print("Starting Allied Vision Camera Stream...")
    print('Starting offboard control node...')
    rclpy.init(args=args)
    drone_data = DroneDataLogger()
    rclpy.spin(drone_data)

    with VmbSystem.get_instance():
        with get_camera(None) as cam:
            setup_camera(cam)

            print("Press <Enter> to stop acquisition.")

            try:
                # Start streaming and process frames
                cam.start_streaming(handler=frame_handler, buffer_count=50)
                input()
            finally:
                cam.stop_streaming()
                rclpy.shutdown()
                #cv2.destroyAllWindows()
    rclpy.shutdown()
    drone_data.destroy_node()
if __name__ == "__main__":
    main()
