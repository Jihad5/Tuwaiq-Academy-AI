import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n-seg.pt')  # You can replace 'yolov8n.pt' with 'yolov8s.pt', 'yolov8m.pt', etc.

# Define some constants for distance estimation
FOCAL_LENGTH = 615  # You may need to adjust this based on your camera's focal length in pixels
KNOWN_WIDTH = 0.5   # Known width of the object in meters (e.g., average shoulder width for a person)

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.65

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
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def main():
    """
    Main function to capture video from the webcam and perform object detection.
    """
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 is the default ID for the webcam

    try:
        while True:
            # Capture frame from the webcam
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(rgb_frame)

            # Extract detection results
            detected_objects = results[0].boxes  # YOLOv8 format

            # Get the frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Loop through all detected objects
            for box in detected_objects:
                # Unpack the bounding box and detection results
                x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                conf = float(box.conf.numpy()[0])     # Confidence score
                class_id = int(box.cls.numpy()[0])    # Class ID
                label = model.names[class_id]         # Class label

                # Check if the confidence is above the threshold
                if conf > CONFIDENCE_THRESHOLD:
                    # Calculate the center of the object
                    center_x, center_y = get_object_center((x1, y1, x2, y2))

                    # Estimate the distance to the object
                    bbox_width = x2 - x1
                    distance_to_object = estimate_distance(bbox_width)
                    print(f"Detected {label} with confidence {conf:.2f}. Estimated distance: {distance_to_object:.2f} meters")

                    # Optionally, draw bounding boxes and labels on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame with annotations
            cv2.imshow('Webcam', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
