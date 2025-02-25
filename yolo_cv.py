import cv2
import torch
from ultralytics import YOLO

# SET THESE MANUALLY
single_eye = False
path = 'runs/detect/train8/weights/best.onnx'

# Load the trained YOLO model
model = YOLO(path)  # Change to your fine-tuned model path

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break  # If no frame is read, exit loop

    # Run YOLO inference on the frame
    results = model(frame)

    # Extract bounding boxes and class names
    boxes = results[0].boxes
    class_names = results[0].names

    # Convert YOLO tensor boxes to NumPy array
    if boxes is not None:
        boxes_data = boxes.xyxy.cpu().numpy()  # Bounding box (x1, y1, x2, y2)
        class_ids = boxes.cls.cpu().numpy()  # Class IDs

        for box, class_id in zip(boxes_data, class_ids):
            class_id = int(class_id)  # Convert class ID to integer
            class_name = class_names.get(class_id, "Unknown")

            # Filter only 'eye' detections
            if class_name == "eye":
                x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                if single_eye:
                    # Calculate the pupil position (center of the bounding box)
                    pupil_x = (x1 + x2) // 2
                    pupil_y = (y1 + y2) // 2
    
                    # Draw a small red dot to mark the pupil
                    cv2.circle(frame, (pupil_x, pupil_y), 2, (0, 0, 255), -1)  # Red dot
                else:
                    l_pupil_x = x1 + (x2 - x1) // 6
                    l_pupil_y = (y1 + y2) // 2
                    r_pupil_x = x1 + 5*(x2 - x1) // 6
                    r_pupil_y = (y1 + y2) // 2
    
                    # Draw a small red dot to mark the pupil
                    cv2.circle(frame, (l_pupil_x, l_pupil_y), 2, (0, 0, 255), -1)  # Red dot                 
                    # Draw a small red dot to mark the pupil
                    cv2.circle(frame, (r_pupil_x, r_pupil_y), 2, (0, 0, 255), -1)  # Red dot                 


    # Show the frame with detections
    cv2.imshow("Live Eye Detection with Pupil Marking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()