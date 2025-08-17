import cv2
import dwani

import os

# Set API key and base URL
dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")

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


import sounddevice as sd
from scipy.io.wavfile import write

def capture_audio(audio_file_name):
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    write(audio_file_name, fs, myrecording)  # Save as WAV file
    print(f"Saved as {audio_file_name}")


if __name__ == "__main__":
    file_name = "photo_description.png"

    # Step 1: Take a picture
    if not take_picture(filename=file_name):
        exit()

    # Step 2: Load and convert image to bytes

    audio_file_name = "input_audio.wav"
    capture_audio(audio_file_name)
    #capture 3 second audio file in wav format and send it here
    asr_result = dwani.ASR.transcribe(file_path=audio_file_name, language="english")

    print(asr_result)
    result = dwani.Vision.caption_direct(
                file_path=file_name,
                query="Describe this image , answer in only one line. Do not explain",
                model="gemma3"
            )
    print("Vision Response: ", result)

