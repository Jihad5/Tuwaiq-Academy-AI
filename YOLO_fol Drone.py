from djitellopy import Tello
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with 'yolov8s.pt', 'yolov8m.pt', etc.

# Define some constants for distance estimation
FOCAL_LENGTH = 615  # You may need to adjust this based on your camera's focal length in pixels
KNOWN_WIDTH = 0.075  # Known width of the object in meters (average width of a cell phone)

# Threshold for stopping distance in meters
STOP_DISTANCE_THRESHOLD = 0.5

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.7

# Drone movement speeds
MOVE_THRESHOLD = 50  # pixels
MAX_SPEED = 30       # cm/s

# Define the timeout for disappearing detection
DISAPPEARANCE_TIMEOUT = 30  # Number of frames before stopping the drone


def estimate_distance(bbox_width):
    """
    Estimate the distance to an object based on the bounding box width.

    :param bbox_width: The width of the bounding box in pixels.
    :return: Estimated distance to the object in meters.
    """
    return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width


def get_object_center(box):
    """
    Calculate the center of a bounding box.

    :param box: The bounding box coordinates (x1, y1, x2, y2).
    :return: The center coordinates (center_x, center_y).
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    return center_x, center_y


def control_drone(center_x, center_y, frame_center_x, frame_center_y, distance_to_object, drone):
    """
    Control the drone's movement to follow the detected object.

    :param center_x: The x-coordinate of the object's center.
    :param center_y: The y-coordinate of the object's center.
    :param frame_center_x: The x-coordinate of the frame's center.
    :param frame_center_y: The y-coordinate of the frame's center.
    :param distance_to_object: The estimated distance to the object in meters.
    :param drone: The Tello drone object.
    """
    # Calculate deltas
    delta_x = center_x - frame_center_x
    delta_y = center_y - frame_center_y

    # Determine speeds based on deltas
    speed_x = 0
    speed_y = 0
    speed_forward = 0

    # Adjust horizontal movement (left/right)
    if abs(delta_x) > MOVE_THRESHOLD:
        # Move right if the object is to the right and vice versa
        speed_x = MAX_SPEED if delta_x > 0 else -MAX_SPEED

    # Adjust vertical movement (up/down)
    if abs(delta_y) > MOVE_THRESHOLD:
        # Move down if the object is above and vice versa
        speed_y = MAX_SPEED if delta_y > 0 else -MAX_SPEED

    # Move forward/backward based on distance
    if distance_to_object > STOP_DISTANCE_THRESHOLD:
        speed_forward = MAX_SPEED  # Move forward if too far
    elif distance_to_object < STOP_DISTANCE_THRESHOLD:
        speed_forward = -MAX_SPEED  # Move backward if too close
    else:
        speed_forward = 0  # Stay still if at the correct distance

    # Apply speeds to the drone
    drone.send_rc_control(0, speed_forward, -speed_y, speed_x)
    print(f"Object detected at ({center_x}, {center_y}). Moving {speed_forward} forward, {speed_x} sideways, {speed_y} vertically.")


def main():
    """
    Main function to control the drone and perform object detection.
    """
    print("Class Names: ", model.names)  # Print the class names

    # Find the class ID for cell phones
    TARGET_CLASS_ID = 0
    for idx, name in enumerate(model.names):
        if name == "cell phone":  # Update the class name to the correct one from model.names
            TARGET_CLASS_ID = idx
            break

    if TARGET_CLASS_ID is None:
        print("Cell phone class not found in model.names")
        return

    print(f"Tracking class ID for cell phone: {TARGET_CLASS_ID}")

    # Initialize the drone
    drone = Tello()

    # Connect to the Tello drone
    drone.connect()

    # Start the video stream
    drone.streamon()

    # Take off
    drone.takeoff()

    # Move up to avoid ground disturbance
    drone.move_up(30)

    # Initialize the disappearance frame counter
    disappearance_counter = 0

    try:
        while True:
            # Capture frame from the drone's camera
            frame = drone.get_frame_read().frame

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(rgb_frame)

            # Extract detection results
            detected_objects = results[0].boxes  # YOLOv8 format

            # Get the frame dimensions
            frame_height, frame_width, _ = frame.shape
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2

            # Flag to check if target object is detected
            target_detected = False

            # Look for the target object in the detections
            for box in detected_objects:
                # Unpack the bounding box and detection results
                x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                conf = float(box.conf.numpy()[0])     # Confidence score
                class_id = int(box.cls.numpy()[0])    # Class ID
                label = model.names[class_id]         # Class label

                # Print confidence score for debugging
                print(f"Detected {label} with confidence {conf:.2f}")

                if conf > CONFIDENCE_THRESHOLD and class_id == TARGET_CLASS_ID:  # Check if confidence is above threshold and is the target class
                    # Calculate the center of the object
                    center_x, center_y = get_object_center((x1, y1, x2, y2))

                    # Estimate the distance to the object
                    bbox_width = x2 - x1
                    distance_to_object = estimate_distance(bbox_width)
                    print(f"Estimated distance: {distance_to_object:.2f} meters")

                    # Control the drone to follow the target
                    control_drone(center_x, center_y, frame_center_x, frame_center_y, distance_to_object, drone)

                    # Reset disappearance counter since target is detected
                    target_detected = True
                    disappearance_counter = 0

                    # Optionally, draw bounding boxes and labels on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Break after finding the first target to reduce processing time
                    break

            # If no target is detected, increase the disappearance counter
            if not target_detected:
                disappearance_counter += 1

            # If disappearance counter exceeds the threshold, stop the drone
            if disappearance_counter > DISAPPEARANCE_TIMEOUT:
                drone.send_rc_control(0, 0, 0, 0)  # Stop all movements
                print("Object not detected. Stopping drone.")

            # Display the frame with annotations
            cv2.imshow('Drone Camera', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Land the drone
        drone.land()
        drone.streamoff()  # Stop video stream
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
