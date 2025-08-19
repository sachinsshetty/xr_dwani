import dwani 
import os 
from playsound import playsound


import sounddevice as sd
from scipy.io.wavfile import write



dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")



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




def step_3():
    print("step3")

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