import dwani 
import os 

dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")

'''
Guests from a neighboring gallaxy finally come to Earth.

You are the delegate of the humanity, chosen to guide our alien friends and answer their questions about our planet.

Put on your headset, equipped with a universal translator and get ready to satisfy their curiosity. 

Legal Aliens is an immersive AR application that will help you learn foreign languages from any objects around you. 

An object recognition system allows you to expand your vocabulary, while an integrated AI prepares follow-up questions and quizes that move your learning even further.
Legal Aliens is not just language studies for humans of all ages, it is about discovering the world around you. 

Open up a portal into the world of adventures. Help the aliens discover the Earth, and explore your favorite places like a local.
'''

def tts_audio(text_to_convert, file_name):
    response = dwani.Audio.speech(input=text_to_convert, response_format="mp3", language="english")


    with open(file_name, "wb") as f:
        f.write(response)



text1 = "Guests from a neighboring gallaxy finally come to Earth."

audio_file_1 = "audio_file_1.mp3" 

tts_audio(text1, audio_file_1)

text2 = "You are the delegate of the humanity, chosen to guide our alien friends and answer their questions about our planet."

audio_file_2 = "audio_file_2.mp3" 

tts_audio(text2, audio_file_2)


text3 = "Put on your headset, equipped with a universal translator and get ready to satisfy their curiosity."

audio_file_3 = "audio_file_3.mp3" 

tts_audio(text3, audio_file_3)


text4 = "Legal Aliens is an immersive AR application that will help you learn foreign languages from any objects around you. "

audio_file_4 = "audio_file_4.mp3" 

tts_audio(text4, audio_file_4)


text5 = "An object recognition system allows you to expand your vocabulary, while an integrated AI prepares follow-up questions and quizes that move your learning even further."
audio_file_5 = "audio_file_5.mp3" 

tts_audio(text5, audio_file_5)


text6 = "Legal Aliens is not just language studies for humans of all ages, it is about discovering the world around you. "

audio_file_6 = "audio_file_6.mp3" 

tts_audio(text6, audio_file_6)


text7 = "Open up a portal into the world of adventures. Help the aliens discover the Earth, and explore your favorite places like a local."

audio_file_7 = "audio_file_7.mp3" 

tts_audio(text7, audio_file_7)
