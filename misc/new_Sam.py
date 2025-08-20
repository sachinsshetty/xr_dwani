import cv2
import numpy as np
from ultralytics.models.fastsam import FastSAMPredictor

# Create FastSAMPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)
predictor = FastSAMPredictor(overrides=overrides)

# Load image path
image_path = "dogs.jpg"
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Segment everything
everything_results = predictor(image_path)

# Prompt inference - returns a list of Results objects
text_results = predictor.prompt(everything_results, texts=["a photo of a dog"])

# Access first Results object from the list
first_result = text_results[0]

# Extract mask tensor from Masks object
mask_tensor = first_result.masks.data

# Convert to CPU and NumPy
masks_np = mask_tensor.cpu().numpy()

# Helper function to overlay masks on the original image with resizing
def overlay_masks(image, masks, alpha=0.5, mask_color=(0, 255, 0)):
    overlay = image.copy()
    height, width = image.shape[:2]
    for mask in masks:
        # Resize mask to original image size
        resized_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        bool_mask = resized_mask.astype(bool)

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[bool_mask] = mask_color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
    return overlay

# Overlay masks on original image
masked_image = overlay_masks(original_image, masks_np)

# Save output image
output_path = "dogs_segmented.png"
cv2.imwrite(output_path, masked_image)
print(f"Segmented image saved to {output_path}")
