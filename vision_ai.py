import cv2
import dwani

import os
from playsound import playsound

import sounddevice as sd
from scipy.io.wavfile import write


import tempfile
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

    tts_response = result['answer']

    response = dwani.Audio.speech(input=tts_response, response_format="wav", language="english")

    tts_filename = "output_english.wav"
    with open(tts_filename, "wb") as f:
        f.write(response)
    print("Audio Speech: Output saved to output_english.wav")

    playsound(tts_filename)

from openai import OpenAI

import os

# Set API key and base URL
tool_server_url = os.getenv("API_SERVER_TOOL")

send_image = False

# Mouse callback function to detect click
def mouse_callback(event, x, y, flags, param):
    global send_image
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        send_image = True


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time for a specified timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA timezone name (e.g., Europe/Berlin)"
                    }
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_webcam_image",
            "description": "Capture a single frame from the default webcam, display it, and return a description on user click",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

import pytz
from datetime import datetime


# Function to get the current time for a given timezone using pytz
def get_current_time(timezone):
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        formatted_time = current_time.strftime("%I:%M:%S %p %Z, %A, %B %d, %Y")
        return {"timezone": timezone, "current_time": formatted_time}
    except pytz.exceptions.UnknownTimeZoneError:
        return {"error": f"Invalid timezone: {timezone}"}
    except Exception as e:
        return {"error": f"Failed to fetch time for {timezone}: {str(e)}"}

# Function to capture, display, and describe a webcam frame
def capture_webcam_image():
    global send_image
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "Failed to open webcam"}

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Failed to capture image from webcam"}

        # Display the captured frame
        cv2.namedWindow("Webcam Frame")
        cv2.setMouseCallback("Webcam Frame", mouse_callback)
        send_image = False  # Reset click flag

        # Show the frame until clicked or 'q' is pressed
        while True:
            cv2.imshow("Webcam Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if send_image or key == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

        if not send_image:
            return {"error": "Image not sent (user closed window without clicking)"}

        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
    except Exception as e:
        return {"error": f"Error capturing webcam image: {str(e)}"}

tool_client = OpenAI(base_url=tool_server_url, api_key="EMPTY")
def tool_call_api():
    pass


import json

def chat_with_qwen3():
    user_prompts = [
        "What do you see?, imagine you are assisting blind person locate all the objects",
        "how should i move from here to the exit?, imagine you are assisting blind person, provide step by step instructions",
    ]
    messages = []

    try:
        for prompt in user_prompts:
            # Add user message
            messages.append({"role": "user", "content": prompt})

            while True:
                response = tool_client.chat.completions.create(
                    model="Qwen3-32B",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=32768
                )
                response_message = response.choices[0].message

                # If there are tool calls, handle them
                if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "get_current_time":
                            args = json.loads(tool_call.function.arguments)
                            timezone = args.get("timezone", "Europe/Berlin")
                            time_data = get_current_time(timezone)
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(time_data),
                                "tool_call_id": tool_call.id
                            })
                        elif tool_call.function.name == "capture_webcam_image":
                            image_data = capture_webcam_image()
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(image_data),
                                "tool_call_id": tool_call.id
                            })
                        else:
                            print(f"Unknown tool called: {tool_call.function.name}")
                    # Continue the loop to let the model use the tool outputs
                else:
                    # No tool calls, print and add the response
                    print(f"Qwen3: {response_message.content}\n")
                    messages.append({"role": "assistant", "content": response_message.content})
                    break

    except Exception as e:
        print(f"Error occurred: {str(e)}")
