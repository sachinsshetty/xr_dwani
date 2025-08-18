from ultralytics import YOLO
import cv2

# Load the YOLOv8n model (pre-trained)
model = YOLO("yolov8n.pt")  # Already downloaded as per your output

# Load an image
image_path = "photo_description.png"  # Ensure this path is correct
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
cv2.imwrite("output.png", image)
cv2.imshow("YOCOv8n Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()