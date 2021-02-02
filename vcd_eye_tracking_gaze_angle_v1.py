#VCD eye tracking and gaze angle

import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1, p2):
	return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
	

#Finds the eye ration (width vs height) using dlib landmarks
#Can be used to identify blinking, eye region centers (not pupil) - very accurate
def get_eye_ratio(eye_points, facial_landmarks):
	left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
	center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

	#draws the line on the eyes
	#hor_line = cv2.line(frame, left_point, right_point, (0, 255,0), 2)
	#ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 2)

	hor_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
	ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

	ratio = hor_line_length/ver_line_length
	return ratio, center_top, center_bottom


#Finds just the eye region given a string of landmarks
def get_eye_region (eye_points, landmarks):
	#window for just the eye
	eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
							(landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
							(landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
							(landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
							(landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
							(landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)
	return eye_region


#Finds just the eye frame from the original frame for viewing/analysis
def get_eye_frame (eye_region, frame):
	min_x = np.min(eye_region[:,0])
	max_x = np.max(eye_region[:,0])
	min_y = np.min(eye_region[:,1])
	max_y = np.max(eye_region[:,1])
	
	eye_height = max_y-min_y+20
	eye_width = max_x-min_x
	
	eye = frame [min_y-10:max_y+10, min_x:max_x]
	eye = cv2.resize(eye, None, fx=5, fy=5)
	return eye, eye_width, eye_height, min_x, min_y


#Finds the iris, and the pupil can be assumed to be in the center of the iris
#Given bad camera quality, pupil in Asian/dark brown eyes are too difficult to differentiate
def get_iris_region (eye, eye_width, eye_height, min_x, min_y):
	grayEye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
	c = cv2.HoughCircles(grayEye, cv2.HOUGH_GRADIENT, 75, 100, minRadius=40, maxRadius = 110)
	
	# ensure at least some circles were found
	if c is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		c = np.round(c[0, :]).astype("int")
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in c:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			eye_x = int(x/5)+min_x
			eye_y = int(y/5)+min_y-10
			if((x+r) <= eye_width*5 and (y+r) <= eye_height*5):
				cv2.circle(frame, (eye_x, eye_y), int(r/5), (255, 255, 0), 2)
				cv2.rectangle(frame, (eye_x - 2, eye_y - 2), (eye_x + 2, eye_y + 2), (0, 128, 255), -1)
				cv2.circle(eye, (x, y-10), r, (255, 255, 0), 4)
				cv2.rectangle(eye, (x - 5, y-10 - 5), (x + 5, y-10 + 5), (0, 128, 255), -1)
				return eye_x, eye_y
	return -1, -1


#Finds the gaze ratio
#For left and right viewing, we can see how much white/black pixels there are
#depending on what side these pixels are on, we can determine which way the user is viewing L/R
#This approach doesn't work for U/D
def get_gaze_ratio(eye_region, facial_landmarks):
	screen_height, screen_width, _ = frame.shape
	mask = np.zeros((screen_height, screen_width), np.uint8)
		
	cv2.polylines(mask, [eye_region], True, 255, 2)
	cv2.fillPoly(mask, [eye_region], 255)
	eye = cv2.bitwise_and(gray, gray, mask=mask)
	
	min_x = np.min(eye_region[:,0])
	max_x = np.max(eye_region[:,0])
	min_y = np.min(eye_region[:,1])
	max_y = np.max(eye_region[:,1])

	gray_eye = eye[min_y:max_y, min_x:max_x]
	
	_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
	
	height, width = threshold_eye.shape
		
	left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
	left_side_white = cv2.countNonZero(left_side_threshold)
		
	right_side_threshold = threshold_eye[0:height, int(width/2):width]
	right_side_white = cv2.countNonZero(right_side_threshold)
	
	if left_side_white == 0:
		gaze_ratio = 0.5
	elif right_side_white == 0:
		gaze_ratio = 5
	else:
		gaze_ratio = left_side_white / right_side_white
	return gaze_ratio
	
	
	
while True:
	_, frame = cap.read()
	height, width, _ = frame.shape
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = detector(gray)
	for face in faces:
		#top left coordinate and bottom right coordinate of face
		#drawing a bounding box on the face
		x0, y0 = face.left(), face.top()
		x1, y1 = face.right(), face.bottom()
		cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
		
		#get the face landmarks using dlib library 68 points
		landmarks = predictor(gray, face)
		
		left_eye_region = get_eye_region([36, 37, 38, 39, 40, 41], landmarks)
		right_eye_region = get_eye_region([42, 43, 44, 45, 46, 47], landmarks)
		
		L_eye, L_eye_width, L_eye_height, L_min_x, L_min_y = get_eye_frame(left_eye_region, frame)
		R_eye, R_eye_width, R_eye_height, R_min_x, R_min_y = get_eye_frame(right_eye_region, frame)
		
		#check for later -> If (x, y) is (-1, -1), then the values are invalid
		L_eye_x, L_eye_y = get_iris_region(L_eye, L_eye_width, L_eye_height, L_min_x, L_min_y)
		R_eye_x, R_eye_y = get_iris_region(R_eye, R_eye_width, R_eye_height, R_min_x, R_min_y)
		
		left_eye_ratio, left_top, left_bottom = get_eye_ratio([36, 37, 38, 39, 40, 41], landmarks)
		right_eye_ratio, right_top, right_bottom = get_eye_ratio([42, 43, 44, 45, 46, 47], landmarks)
		
		#LEFT AND RIGHT GAZE ANGLE
		gaze_ratio_left_eye = get_gaze_ratio(left_eye_region, landmarks)
		gaze_ratio_right_eye = get_gaze_ratio(right_eye_region, landmarks)
		average_gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye)/2

		if average_gaze_ratio <= 0.6:
			cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
		elif 0.6 < average_gaze_ratio < 2.3:
			cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
		else:
			cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
			
		#UP AND DOWN GAZE ANGLE
		if L_eye_y != -1 and R_eye_y != -1:
			eye_y = int((L_eye_y + R_eye_y)/2)
		elif L_eye_y != -1:
			eye_y = L_eye_y
		else:
			eye_y = R_eye_y;
			
		if eye_y != -1:
			if eye_y < int(height/2)-10:
				cv2.putText(frame, "DOWN", (50, 300), font, 2, (0, 0, 255), 3)
			elif eye_y > int(height/2) + 10:
				cv2.putText(frame, "UP", (50, 300), font, 2, (0, 0, 255), 3)
			else:
				cv2.putText(frame, "CENTER", (50, 300), font, 2, (0, 0, 255), 3)
		
		#cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
		cv2.putText(frame, str(average_gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
		
		cv2.imshow("Left Eye", L_eye)
		cv2.imshow("Right Eye", R_eye)
		
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
