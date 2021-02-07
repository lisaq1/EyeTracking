from __future__ import print_function
import cv2 as cv
import math
import argparse

#constants used in gaze angle calculations
#~3.80 pixels to 1 mm
pixel_conversion = 3.8
#ideal distance for computer screen users
z_distance = 400


#Identify/Detect the face
def detectAndDisplay(frame):
	#Simplify the image (color isn't really necessary)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#graphical rep of the gray picture
	frame_gray = cv.equalizeHist(frame_gray)
	height, width, _ = frame.shape
    
    #-- Detect faces
	faces = face_cascade.detectMultiScale(frame_gray)
	
	left_eye = None
	right_eye = None
	pupilliary_distance = None
	left_pitch = None
	left_yaw = None
	right_pitch = None
	right_yaw = None
	
    #Look through each face on the screen
	for (x,y,w,h) in faces:
		center = (x + w//2, y + h//2)
        #bounding the face frame
		frame = cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 4)
        #frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
		faceROI = frame_gray[y:y+h,x:x+w]
		
        #-- In each face, detect eyes
		eyes = eyes_cascade.detectMultiScale(faceROI)
		for (ex,ey,ew,eh) in eyes:
			#filtering potential false eyes (mouth and nose)
			if ey+eh < int(2*h/3):
				#create eye region
				eyeROI = frame[ey+y:ey+eh+y, ex+x:ex+ew+x]
				eyeROI_center = (x + ex + ew//2, y + ey + eh//2)
				radius = int(round((ew + eh)*0.25))
				#eye drawing
				#frame = cv.circle(frame, eyeROI_center, radius, (255, 0, 0 ), 4)
				x_pos, y_pos = get_iris_region(eyeROI, x, y, ex, ey, frame)
				if x_pos < x + (w//2) and x_pos != -1:
					left_eye = (x_pos, y_pos)
				elif x_pos != -1:
					right_eye = (x_pos, y_pos)
				
				if left_eye != None and right_eye != None:
					pupilliary_distance = get_pupil_distance(left_eye, right_eye)
					cv.line(frame, left_eye, right_eye, (0, 0, 255), 2)
					left_pitch, left_yaw = get_pitch_yaw_angle(left_eye, height, width)
					right_pitch, right_yaw = get_pitch_yaw_angle(right_eye, height, width)
			
					print("left_eye: (", math.degrees(left_pitch), math.degrees(left_yaw), ")")
					print("right_eye: (", math.degrees(right_pitch), math.degrees(right_yaw), ")")

	cv.imshow('Capture - Face detection', frame)


def get_iris_region(eyeROI, face_x, face_y, eye_x, eye_y, frame):
	grayEye = cv.cvtColor(eyeROI, cv.COLOR_BGR2GRAY)
	grayEye = cv.GaussianBlur(grayEye, (7, 7), 0)
	_, threshold = cv.threshold(grayEye, 30, 255, cv.THRESH_BINARY_INV)
	contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse = True)
	
	for cnt in contours:
		(x, y, w, h) = cv.boundingRect(cnt)
		x_pos = x+int(w/2)+face_x+eye_x
		y_pos = y+int(h/2)+face_y+eye_y
		cv.circle(frame, (x_pos, y_pos), 5, (255, 255, 255), 3)
		return x_pos, y_pos
	
	return -1, -1
	

def get_pupil_distance(left, right):
	return right[0] - left[0]


def get_pitch_yaw_angle(eye, screen_height, screen_width):
	#if
	pitch_angle = math.atanh(abs(eye[1] - int(screen_height/2)) / pixel_conversion / z_distance)
	yaw_angle = math.atanh(abs(eye[0] - int(screen_width/2)) / pixel_conversion / z_distance)
	return pitch_angle, yaw_angle


#Defining Haar Cascade classifier (eyes and face)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)


args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
    

while True:
    ret, frame = cap.read()
    
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
        
    detectAndDisplay(frame)
    
    if cv.waitKey(10) == 27:
        break
