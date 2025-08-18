import torch
from PIL import Image
import cv2
import numpy as np
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the model and processor
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

# Load image
image_path = "sample_image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Define text prompt
text_prompt = "dog . cat . person"

# Process inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,  # Confidence threshold
    text_threshold=0.3,  # Text confidence threshold
    target_sizes=[image.size[::-1]]  # (height, width)
)[0]

# Visualize results using supervision
detections = sv.Detections(
    xyxy=results["boxes"].cpu().numpy(),
    confidence=results["scores"].cpu().numpy(),
    class_id=np.array([0] * len(results["scores"])),  # Dummy class IDs
    labels=results["labels"]
)

# Annotate image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Save or display the result
cv2.imwrite("annotated_image.jpg", annotated_image[:, :, ::-1])  # Save as BGR
# To display: cv2.imshow("Result", annotated_image[:, :, ::-1]); cv2.waitKey(0); cv2.destroyAllWindows()