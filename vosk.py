from gtts import gTTS
import os
import tempfile
import requests
import json
from vosk import Model, KaldiRecognizer

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
    vosk_api_url = "https://api.vosk.ai/asr/online"
    audio_file = "/path/to/your/audio/file.wav"  # Replace with the actual path to your audio file

    model_path = "/path/to/your/vosk-model-en-us-aspire-0.2"  # Replace with the actual path to your Vosk model
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with open(audio_file, "rb") as audio_data:
        files = {"file": audio_data}
        response = requests.post(vosk_api_url, files=files)

        if response.status_code == 200:
            result = json.loads(response.text)
            text = result["result"]
            return text.lower()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

# Example usage:
# text_input = get_voice_input()
# print(f"User said: {text_input}")
