# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:12:32 2020

@author: nagar
"""

import numpy as np
import cv2
import time
import math

cam = cv2.VideoCapture(0)

time.sleep(1)
num_frames = 0
bg = None
aWeight = 0.8

def init_bgnd():
    global bg
    for i in range(30): 
        return_val, bg = cam.read() 
        if return_val == False : 
            continue 
  
    bg = np.flip(bg, axis = 1)          # .astype('float') # flipping of the frame 
    return bg


bg = init_bgnd()

while True:
    ret,frame = cam.read()
    if ret is False:
        pass
    
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()            # .astype('float')
    
    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 174, 223])        
    upper_red = np.array([100, 255, 255]) 
    mask1 = cv2.inRange(hsv, lower_red, upper_red) 
    # setting the lower and upper range for mask2  
    #cv2.imshow('mask 1',mask1)
    
    lower_red = np.array([155, 40, 40]) 
    upper_red = np.array([180, 255, 255]) 
    mask2 = cv2.inRange(hsv, lower_red, upper_red) 
    
    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated
    
    mask1 = mask1 + mask2 
  
    # Refining the mask corresponding to the detected red color 
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), 
                                         np.uint8), iterations = 2) 
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 1) 
    #cv2.imshow('Mask 1 after morph',mask1)
    mask2 = cv2.bitwise_not(mask1) 
    #cv2.imshow('mask 2', mask2)
    #print(mask2)
    # Generating the final output 
    res1 = cv2.bitwise_and(bg, bg, mask = mask1) 
    res2 = cv2.bitwise_and(clone, clone, mask = mask2) 
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) 
    cv2.imshow("Invisible", final_output) 
  
    #dynamic_cam = static_background()
    
    #static_cam(frame,gray,good_new,good_old)
    
    cv2.imshow('my_frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
    