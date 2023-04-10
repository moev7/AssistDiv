import pygame
import time

pygame.mixer.init()

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
