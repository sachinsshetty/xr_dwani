import torch
from PIL import Image
import cv2
import numpy as np
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and processor
model_id = "IDEA-Research/grounding-dino-tiny"
try:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# Load image
image_path = "input_data.jpeg"  # Replace with your image path
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()
image_np = np.array(image)

# Define text prompt
text_prompt = "laptop . person"

# Process inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results using the correct method
try:
    results = processor.post_process_grounding_dino(
        outputs,
        inputs=inputs,
        threshold=0.35,  # confidence threshold
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]
except Exception as e:
    print(f"Error in post-processing: {e}")
    exit()

# Extract detections
boxes = results["boxes"].cpu().numpy()
scores = results["scores"].cpu().numpy()
labels = results["labels"]

# Filter detections based on confidence (redundant, threshold applied above, but kept for illustration)
min_confidence = 0.35
mask = scores >= min_confidence
boxes = boxes[mask]
scores = scores[mask]
labels = labels[mask]

# Map label IDs to text labels
label_map = {i: label for i, label in enumerate(text_prompt.split(" . "))}
str_labels = [label_map.get(int(label), "unknown") for label in labels]

# Visualize results using supervision
detections = sv.Detections(
    xyxy=boxes,
    confidence=scores,
    class_id=np.array([0] * len(scores)),  # Dummy class IDs
    labels=str_labels
)

# Annotate image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Save the result
output_path = "annotated_image.jpg"
cv2.imwrite(output_path, annotated_image[:, :, ::-1])  # Save in BGR format for OpenCV
print(f"Annotated image saved as {output_path}")

# Optional: Display the result
# cv2.imshow("Result", annotated_image[:, :, ::-1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
