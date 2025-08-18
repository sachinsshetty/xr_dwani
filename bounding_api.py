from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#model = YOLO("yolov8n.pt")
model = YOLO("yolov8l.pt")  # example for large model with better accuracy


def read_imagefile(file) -> np.ndarray:
    image_bytes = file.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image
from typing import List, Optional
from fastapi import Query


@app.post("/detect/")
async def detect(
    image_file: UploadFile = File(...),
    label_filter: Optional[List[str]] = Query(None, description="Filter to only include detections matching any of these labels"),
    confidence_threshold: float = Query(0.9, gt=0, lt=1, description="Minimum confidence threshold for detections"),
    top_k: int = Query(5, gt=0, description="Maximum number of top detections to return, sorted by confidence")
):
    image = read_imagefile(image_file)
    results = model(image)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf.cpu().numpy().item())
            class_id = int(box.cls.cpu().numpy().item())
            label = model.names[class_id]
            if confidence >= confidence_threshold:
                if label_filter is None or any(f.lower() in label.lower() for f in label_filter):
                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "class_id": class_id,
                        "label": label
                    })
    # Sort detections by confidence, descending, and limit with top_k
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:top_k]

    return {"detections": detections}


@app.post("/detect-image/")
async def detect_image(
    image_file: UploadFile = File(...),
    label_filter: Optional[List[str]] = Query(None, description="Filter to only show labels containing any of these strings"),
    confidence_threshold: float = Query(0.9, gt=0, lt=1, description="Minimum confidence threshold for detections"),
    top_k: int = Query(5, gt=0, description="Maximum number of top detections to draw, sorted by confidence")
):
    image = read_imagefile(image_file)
    results = model(image)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf.cpu().numpy().item())
            class_id = int(box.cls.cpu().numpy().item())
            label = model.names[class_id]
            if confidence >= confidence_threshold:
                if label_filter is None or any(f.lower() in label.lower() for f in label_filter):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "label": label
                    })

    # Sort and keep only top_k by confidence
    top_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:top_k]

    # Draw boxes only for top detections
    for det in top_detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        confidence = det["confidence"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', image)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
