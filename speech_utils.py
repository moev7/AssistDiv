from vosk import Model, KaldiRecognizer
from gtts import gTTS
import os
import tempfile
import speech_recognition as sr
import requests
import json


def speak(text, language='en', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        temp_file = fp.name + ".mp3"
        tts.save(temp_file)
        os.system(f"mpg123 {temp_file}")


def announce_objects(detected_objects):
    if not detected_objects:
        speak("No objects detected.")
    else:
        obj_count = len(detected_objects)
        speak(f"Detected {obj_count} objects.")
        for obj in detected_objects:
            speak(f"{obj['name']} at {obj['distance']:.2f} meters.")


def get_voice_input():
    r = sr.Recognizer()
    model = Model(json.loads(requests.get("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip").content))
    recognizer = KaldiRecognizer(model, 16000)

    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            # Use Vosk for speech recognition
            text = r.recognize_vosk(audio, recognizer)
            return text.lower()
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Vosk service; {e}")
            return None


# Example usage:
# text_input = get_voice_input()
# if text_input:
#     print(f"Recognized text: {text_input}")
