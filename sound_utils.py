import pygame
import time
import pyrealsense2.pyrealsense2 as rs
import numpy as np
from language_actions import language_actions
from speech_utils import speak
pygame.mixer.init()

BEEP_SOUND_FILE = "bleep-41488.mp3"
DISTANCE_THRESHOLD = 500
DEPTH_TO_CM_CONVERSION = 10

def play_beep_sound(updated_distance, language):
    beep = pygame.mixer.Sound(BEEP_SOUND_FILE)
    # Determine the play duration based on the object's distance
    if updated_distance > 2:
        play_duration = 0.25 
    elif updated_distance > 1.5:
        play_duration = 0.5 
    elif updated_distance > 1:
        play_duration = 1 
    elif updated_distance < 1:
        play_duration = 3
        speak(language_actions[language]['stop'], language) 

    # Play the beep sound with the specified duration
    beep.play(-1, 0, int(play_duration * 1000))  # -1 means loop indefinitely, 0 means no start time

    # Sleep for the duration of playtime
    pygame.time.wait(int(play_duration * 1000))

    # Stop the sound
    beep.stop()

def play_obstacle_beep_sound(depth_image):
    beep = pygame.mixer.Sound(BEEP_SOUND_FILE)
    beep_playing = False

    try:
        while True:
            if np.any(depth_image < DISTANCE_THRESHOLD / DEPTH_TO_CM_CONVERSION):
                if not beep_playing:
                    beep.play()
                    beep_playing = True
            else:
                if beep_playing:
                    pygame.mixer.stop()
                    beep_playing = False
    finally:
        pygame.mixer.quit()

# Add any necessary code to run the functions or perform other actions based on your use case.
