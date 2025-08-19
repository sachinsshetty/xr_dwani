import dwani 
import os 
from playsound import playsound

import cv2
import sounddevice as sd
from scipy.io.wavfile import write



dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")


import sounddevice as sd


def capture_audio(audio_file_name):
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    write(audio_file_name, fs, myrecording)  # Save as WAV file
    print(f"Saved as {audio_file_name}")

def step_1():
    print("step1")

    tts_response = "Who dares enter my kingdom"

    response = dwani.Audio.speech(input=tts_response, response_format="wav", language="english")

    tts_filename = "output_english.wav"
    with open(tts_filename, "wb") as f:
        f.write(response)
    #print("Audio Speech: Output saved to output_english.wav")

    playsound(tts_filename)


def step_2():
    print("step2")

    audio_file_name = "input_audio.wav"
    capture_audio(audio_file_name)
    #capture 3 second audio file in wav format and send it here
    asr_result = dwani.ASR.transcribe(file_path=audio_file_name, language="english")

    print(asr_result)



import requests

def send_image_get_modified(image_path, confidence_threshold=0.7, top_k=3):
    url = 'https://api.dwani.ai/detect-image/'
    params = {
        'confidence_threshold': confidence_threshold,
        'top_k': top_k
    }
    files = {
        'image_file': (image_path, open(image_path, 'rb'), 'image/jpeg')
    }
    headers = {
        'accept': 'application/json'
    }

    response = requests.post(url, params=params, headers=headers, files=files)
    
    if response.status_code == 200:
        # Assuming the API returns the modified image content directly
        return response.content
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


import requests

def send_image_get_detection(image_path, confidence_threshold=0.7, top_k=3):
    url = 'https://api.dwani.ai/detect/'
    params = {
        'confidence_threshold': confidence_threshold,
        'top_k': top_k
    }
    files = {
        'image_file': (image_path, open(image_path, 'rb'), 'image/jpeg')
    }
    headers = {
        'accept': 'application/json'
    }

    response = requests.post(url, params=params, headers=headers, files=files)

    if response.status_code == 200:
        detection_data = response.json()
        # Extract labels from the detection data
        labels = [item['label'] for item in detection_data.get('predictions', [])]
        return labels
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    

def step_3():
    print("step3")

    image_file_name = " image_capture.png"

    take_picture(image_file_name)

    
    modified_image = send_image_get_modified(image_file_name)
    with open('modified_image.jpg', 'wb') as f:
         f.write(modified_image)


    detection_result = send_image_get_detection(image_file_name)
    print(detection_result)




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



def step_4():
    print("step4")

def step_5():
    print("step5")

def step_6():
    print("step6")

def step_7():
    print("step7")

if __name__ == "__main__":
    step_1()
    step_2()
    step_3()
    step_4()
    step_5()
    step_6()
    step_7()