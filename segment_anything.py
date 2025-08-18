import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# Set up the SAM model
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

# Load and preprocess the image
IMAGE_PATH = "dog.png"  # Replace with your image path
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image for the predictor
mask_predictor.set_image(image_rgb)

# Define the bounding box prompt [x_min, y_min, x_max, y_max]
bbox_prompt = np.array([100, 100, 300, 300])  # Adjust coordinates based on your image

# Predict the mask
masks, scores, logits = mask_predictor.predict(
    box=bbox_prompt,
    multimask_output=False
)

# Extract bounding box from the mask
mask = masks[0]  # Use the first mask
y, x = np.where(mask)  # Get coordinates of non-zero mask pixels
if len(x) > 0 and len(y) > 0:
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    detected_bbox = [x_min, y_min, x_max, y_max]
else:
    detected_bbox = None

# Visualization function
def show_mask_and_bbox(image, mask, bbox, detected_bbox):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if mask is not None:
        plt.imshow(mask, alpha=0.5, cmap="jet")  # Overlay mask
    if bbox is not None:
        plt.gca().add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            fill=False, edgecolor="green", linewidth=2
        ))
    if detected_bbox is not None:
        plt.gca().add_patch(plt.Rectangle(
            (detected_bbox[0], detected_bbox[1]), detected_bbox[2] - detected_bbox[0], detected_bbox[3] - detected_bbox[1],
            fill=False, edgecolor="blue", linewidth=2
        ))
    plt.axis("off")
    plt.show()

# Visualize the result
show_mask_and_bbox(image_rgb, masks[0], bbox_prompt, detected_bbox)

# Print the detected bounding box
if detected_bbox:
    print(f"Detected Bounding Box: [x_min: {detected_bbox[0]}, y_min: {detected_bbox[1]}, x_max: {detected_bbox[2]}, y_max: {detected_bbox[3]}]")
else:
    print("No object detected within the provided bounding box.")