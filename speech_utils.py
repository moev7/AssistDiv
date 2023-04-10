from gtts import gTTS
import os
import tempfile


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