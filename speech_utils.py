from gtts import gTTS
import json
import os
import tempfile
from vosk import Model, KaldiRecognizer
import pyaudio
import pyttsx3
import pygame

def play_beep():
    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound('beep_sound.mp3')  
    beep_sound.play()
    pygame.time.wait(int(beep_sound.get_length() * 100)) 


def speak(text, language='en', slow=False):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100 if slow else 130)  
    engine.setProperty('volume', 1.0)  # Adjust the volume (0.0 to 1.0)

    voices = engine.getProperty('voices')
    language_voices = [voice for voice in voices if language in voice.id or language in voice.name]
    if language_voices:
        engine.setProperty('voice', language_voices[0].id)
    else:
        print(f"No voice found for language: {language}")

    engine.say(text)
    engine.runAndWait()


# def speak(text, language='en', slow=False):
#     tts = gTTS(text=text, lang=language, slow=slow)
    
#     with tempfile.NamedTemporaryFile(delete=True) as fp:
#         temp_file = fp.name + ".mp3"
#         tts.save(temp_file)
#         os.system(f"mpg123 {temp_file}")

def announce_objects(detected_objects):
    if not detected_objects:
        speak("No objects detected.")
    else:
        obj_count = len(detected_objects)
        speak(f"Detected {obj_count} objects.")
        for obj in detected_objects:
            speak(f"{obj['name']} at {obj['distance']:.2f} meters.")

def get_voice_input(language = 'en'):
    # Load the Vosk model
    script_directory = os.path.dirname(os.path.realpath(__file__))
    if language == 'en':
        vosk_model_path = os.path.join(script_directory, "vosk-model-small-en-us-0.15")
    elif language == 'es':
        vosk_model_path = os.path.join(script_directory, "vosk-model-small-es-0.42")
    model = Model(vosk_model_path)

    play_beep()


    # Create a recognizer
    rec = KaldiRecognizer(model, 16000)  # assuming 16kHz sampling rate

    # Start listening
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    #TODO: play the listening beep
    print("Listening...")

    while True:
        data = stream.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result:
                # print(result['text'])  # print the recognized text
                return result['text']

    # If nothing was recognized
    speak("Sorry, I did not understand that. Please try again.")
    return None

# import time
# from gtts import gTTS
# import os
# import tempfile
# import speech_recognition as sr

# def speak(text, language='en', slow=False):
#     tts = gTTS(text=text, lang=language, slow=slow)
    
#     with tempfile.NamedTemporaryFile(delete=True) as fp:
#         temp_file = fp.name + ".mp3"
#         tts.save(temp_file)
#         os.system(f"mpg123 {temp_file}")

# def announce_objects(detected_objects):
#     if not detected_objects:
#         speak("No objects detected.")
#     else:
#         obj_count = len(detected_objects)
#         speak(f"Detected {obj_count} objects.")
#         for obj in detected_objects:
#             speak(f"{obj['name']} at {obj['distance']:.2f} meters.")

# def get_voice_input(language='en'):
#     max_attempts = 3  # Set the maximum number of attempts
#     for attempt in range(1, max_attempts + 1):
#         print(f"Attempt {attempt}: Please speak within 5 seconds.")
#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             print("listening...")
#             audio = r.listen(source, timeout=5, phrase_time_limit=5)
#             try:
#                 text = r.recognize_google(audio, language=language).lower()  # Using Google's voice recognition with specified language
#                 return text
#             except sr.UnknownValueError:
#                 speak("Sorry, I did not understand that. Please try again.")
#             except sr.RequestError as e:
#                 speak(f"Could not request results from Google Speech Recognition service; {e}")
#                 return None
#         time.sleep(1)  # Wait for a second before the next attempt
#     speak(f"Exceeded maximum attempts. Exiting.")
#     return None

