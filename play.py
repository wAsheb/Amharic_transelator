# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:48:02 2018

@author: e10509
"""

#import sys
#
#sys.dont_write_bytecode = True
#PYTHONDONTWRITEBYTECODE = 1
#
#import vlc
#
## p = vlc.MediaPlayer("./exclamation.mp3")
#p = vlc.MediaPlayer("./bizunesh.mp3")
#p.play()
#p.play()

import pygame

def playSound(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

def main():
#    playSound("./exclamation.mp3")
    playSound("./bizunesh.mp3")


if __name__ == "__main__":
    main()