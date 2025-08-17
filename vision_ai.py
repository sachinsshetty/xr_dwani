import cv2
from io import BytesIO
from openai import OpenAI
import base64
from PIL import Image
import io


def get_openai_client(model: str) -> OpenAI:
    """Initialize OpenAI client with model-specific base URL."""
    valid_models = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")

    model_ports = {
        "qwen3": "9100",
        "gemma3": "9000",
        "moondream": "7882",
        "qwen2.5vl": "7883",
        "sarvam-m": "7884",
        "deepseek-r1": "7885"
    }
    base_url = f"http://0.0.0.0:{model_ports[model]}/v1"

    return OpenAI(api_key="http", base_url=base_url)


def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def ocr_page_with_rolm(img_base64: str, model: str) -> str:
    """Perform OCR on the provided base64 image using the specified model."""
    try:
        client = get_openai_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {"type": "text", "text": "Return the plain text extracted from this image."}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        # return the text output
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"OCR processing failed: {str(e)}")


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


if __name__ == "__main__":
    file_name = "photo_description.png"

    # Step 1: Take a picture
    if not take_picture(filename=file_name):
        exit()

    # Step 2: Load and convert image to bytes
    image = Image.open(file_name)  
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Step 3: Encode to base64
    encoded_img = encode_image(img_bytes)

    # Step 4: Perform OCR using chosen model
    model_name = "qwen3"  # change to any supported vision model
    result_text = ocr_page_with_rolm(encoded_img, model=model_name)

    # Step 5: Print result
    print("\n--- OCR Extracted Text ---\n")
    print(result_text)
