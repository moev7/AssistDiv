import pygame
import time
import pyrealsense2.pyrealsense2 as rs
import numpy as np


pygame.mixer.init()
'''
def play_beep_sound(updated_distance):
    beep = pygame.mixer.Sound("beep-07a.wav")
    if updated_distance > 2:
        interval = 2.0  # 1 second
    elif updated_distance > 1:
        interval = 0.6  # 600 milliseconds
    elif updated_distance > 0.5:
        interval = 0.3  # 300 milliseconds
    else:
        interval = 0.1  # 100 milliseconds

    beep.play()
    time.sleep(interval)
'''

def play_beep_sound(updated_distance):
    pygame.mixer.init()
    beep = pygame.mixer.Sound("beep-07a.wav")

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
    '''
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    pipeline.start(config)
    '''
    # Initialize Pygame
    pygame.mixer.init()
    beep = pygame.mixer.Sound("beep-07a.wav")
    beep_playing = False

    try:
        while True:        
            if np.any(depth_image < 500):  # 500 corresponds to 50 cm (1 cm = 10 mm)
                if not beep_playing:
                    beep.play()
                    beep_playing = True
            else:
                if beep_playing:
                    pygame.mixer.stop()
                    beep_playing = False
    finally:
        pygame.mixer.quit()
