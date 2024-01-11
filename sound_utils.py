import pygame
import time
import pyrealsense2.pyrealsense2 as rs
import numpy as np

pygame.mixer.init()

BEEP_SOUND_FILE = "beep-07a.wav"
DISTANCE_THRESHOLD = 500
DEPTH_TO_CM_CONVERSION = 10

def play_beep_sound(updated_distance):
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

    # Play the beep sound in a loop
    beep.play(-1)  # the -1 means to loop indefinitely

    # Sleep for the duration of playtime, then stop the sound
    time.sleep(play_duration)
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
