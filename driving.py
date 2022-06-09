from PIL import ImageGrab
import numpy as np
import cv2
import math
import directKeys as keys
import time

#keep track of frames (used when determining if stuck on a wall)
frame = 0
#acculumator of frames to determine when to reverse (stuck at wall when reverseVar > 5 - stuck for 5 frames)
reverseVar = 0
#accumulator of freames to determine sharper turn (hold turn direction longer)
turnVar = 0
#red HSV values for filtering the road
RED_MIN = np.array([165, 0, 0],np.uint8)
RED_MAX = np.array([190, 255, 255],np.uint8)
#blue HSV values for filtering the road
BLUE_MIN = np.array([105, 0, 0],np.uint8)
BLUE_MAX = np.array([120, 255, 255],np.uint8)

#function to mask edge map to reduce edge noise
def region(image):
    height, width = image.shape
    #the spacial difference between these two triangles(masks) will be what hough line detection will be ran on
    triangle = np.array([
                       [(-1000, 400), (400, 167), (width+1000, 400)]
                       ])
    # triangle2 = np.array([
    #                    [(-400, 400), (400, 185), (width+400, 400)]
    #                    ])
    #create both masks
    mask = np.zeros_like(image)
    mask2 = np.zeros_like(image)
    mask2.fill(255)
    mask = cv2.fillPoly(mask, triangle, 255)
    # mask2 = cv2.fillPoly(mask2, triangle2, 0)
    #bitwise operators to get desired space in edge map
    mask = cv2.bitwise_and(image, mask)
    # mask = cv2.bitwise_and(mask, mask2)
    return mask

#function tou turn in 
def reverse(direction):
    keys.ReleaseKey(0x2D)
    time.sleep(.5)
    keys.PressKey(0x2C)
    keys.PressKey(direction)
    time.sleep(0.75)
    keys.ReleaseKey(0x2C)
    keys.ReleaseKey(direction)
    time.sleep(.5)
    keys.PressKey(0x2D)
    return

def turn(direction):
    keys.PressKey(direction)
    time.sleep(0.1)
    keys.ReleaseKey(direction)
    time.sleep(0.1)
    return

try:
    while(True):
        
        #window assumed to be snapped to top right on 1440p monitor
        #can change to full screen if using multiple monitors
        img = np.array(ImageGrab.grab(bbox=(1520, 32, 2320, 700)))
        cv_img = img.astype(np.uint8)
        
        #apply gaussian blur
        kernel = 5
        imgBlur = cv2.GaussianBlur(cv_img,(kernel, kernel), 0)
        
        #convert to HSV
        imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
        
        #filter images to just contain roads
        imgRedThresh = cv2.inRange(imgHSV, RED_MIN, RED_MAX)
        imgBlueThresh = cv2.inRange(imgHSV, BLUE_MIN, BLUE_MAX)
        imgThresh = cv2.add(imgRedThresh, imgBlueThresh)
        
        #dilate image to reduce some noise
        kernel = np.ones((10, 10),np.uint8)
        dilated = cv2.dilate(imgThresh,kernel,iterations = 1)
        
        #apply canny edge detection
        low = 50
        high = 150
        edges = cv2.Canny(dilated, low, high)
        
        #get desired area of edge map
        edges = region(edges)
        
        
        # cv2.imshow("test", edges)
        # cv2.waitKey(0)
        
        #apply hough line detection on masked edge map
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, np.array([]), minLineLength=100, maxLineGap=20)

        #variable to store avg angles of lines computed
        avg = 0
        
        #variable to store increments of unit circle occupied
        count = 0
        
        #variable to store number of lines detected
        numLines = 0
        
        #array corresponding to 5 degree increments in unit circle (angles[0] = 0 < angle < 5)
        angles = [None] * 36
        
        #draw lines on original imagex
        if(lines is not None):
            for line in lines:
                #get start/end coords of line
                x1, y1, x2, y2 = line.reshape(4)
        
                #calculate angle of line
                dx = x1-x2
                dy = y2-y1
                angleR = math.atan(dy/dx)
                angle = math.degrees(angleR)
                if(angle < 0):
                    angle = 180 + angle
                #check if index of angles is occupied
                index = math.floor(angle/5)
                if((angles[index] is None) and ((angle > 10 and angle < 80) or (angle > 100 and angle < 170))):
                    angles[index] = angle
                    avg += angle
                    count += 1
                    numLines += 1
                    cv2.line(cv_img,(x1,y1), (x2,y2), (0,0,255), 2)
            try:
                avg = avg / count
            except Exception:
                avg = 0
            print(avg)

        keys.PressKey(0x2D)
        if(numLines < 1):
            reverseVar += 1
            if(reverseVar > 10):
                if(avg < 90):
                    direction = 0xCD
                else:
                    direction = 0xCB
                if(numLines > 0):
                    reverse(direction)
                    reverseVar = 0
                else:
                    reverse(direction)
                    reverseVar = 0
            else:
                if(avg < 70):
                    turn(0xCD)
                elif(avg > 110):
                    turn(0xCB)
                else:
                    keys.PressKey(0x2D)
                    time.sleep(0.5)
                    keys.ReleaseKey(0x2D)
        else:
            reverseVar = 0
            if(avg < 70):
                turn(0xCD)
            elif(avg > 110):
                turn(0xCB)
            else:
                keys.PressKey(0x2D)
                time.sleep(0.5)
                keys.ReleaseKey(0x2D)
                
        #show original image with hough lines drawn
        cv2.imshow("test", cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite("output.jpg", cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        #cv2.imshow('test', edges)
        cv2.waitKey(0)
        
        frame += 1
        
#handle ending of program
except KeyboardInterrupt:
    pass    
    cv2.destroyAllWindows()
    
