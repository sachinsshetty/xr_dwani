from ultralytics import YOLO
import cv2



# Load the YOLOv8n model (pre-trained)
model = YOLO("yolov8n.pt")  # Already downloaded as per your output



def take_picture(filename: str) -> bool:
    """Capture a single image from the webcam."""
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return False

    ret, frame = cam.read()

    if ret:
        cv2.imwrite(filename, frame)
        print(f"Photo captured and saved as {filename}")
    else:
        print("Failed to capture image")

    cam.release()
    return ret



def get_bounding_box(filename):
    # Load an image
    image_path = filename  # Ensure this path is correct
    image = cv2.imread(image_path)

    # Check if image loaded successfully
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Perform object detection
    results = model(image)


    return results
'''
    # Process results
    for result in results:
        boxes = result.boxes  # Bounding box coordinates
        for box in boxes:
            # Extract coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Move to CPU before NumPy conversion
            confidence = box.conf.cpu().numpy().item()  # Extract scalar value
            class_id = int(box.cls.cpu())  # Move to CPU
            label = model.names[class_id]  # Class name

            # Draw bounding box and label on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite("output.jpeg", image)
    #cv2.imshow("YOCOv8n Detection", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return results 
'''

def get_bounding_box_image(filename):
    image_path = filename  # Ensure this path is correct
    image = cv2.imread(image_path)

    # Check if image loaded successfully
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Perform object detection
    results = model(image)


    # Process results
    for result in results:
        boxes = result.boxes  # Bounding box coordinates
        for box in boxes:
            # Extract coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Move to CPU before NumPy conversion
            confidence = box.conf.cpu().numpy().item()  # Extract scalar value
            class_id = int(box.cls.cpu())  # Move to CPU
            label = model.names[class_id]  # Class name

            # Draw bounding box and label on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite("output.jpeg", image)
    cv2.imshow("YOCOv8n Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results 


file_name = "input_data.jpeg"


take_picture(filename=file_name)
boxes = get_bounding_box(filename=file_name)

print(boxes)


bounding_box_image = get_bounding_box_image(file_name)

