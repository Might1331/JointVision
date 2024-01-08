import cv2
import mediapipe as mp
from prettytable import PrettyTable
from collections import deque
import keyboard
import math
import cmath
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
VISIBILITY_THRESHOLD = 0.95
STANDARD_DEVIATION_THRESHOLD = 15 #degrees
LENGTH_OF_QUEUE = 100 #n

# get_angle joins the points p0, p2 and p1, p3 and then find the angle between them in degrees. Angle less than 180 degree will be returned.
def get_angle(p0, p1, p2, p3):
  v1 = complex(p2.x-p0.x, p2.y-p0.y)
  v2 = complex(p3.x-p1.x, p3.y-p1.y)
  angle = abs(math.degrees(cmath.phase(v1))-math.degrees(cmath.phase(v2)))
  if angle>180:
      angle = 360-angle
  return angle

# def is_key_pressed(c):
#     key = cv2.waitKey(1)
#     if(key == ord('r')):
#         return True
#     else:
#         return False

def get_left_elbow_angle(prev_n_landmarks_left):
    detected = 0
    angle_sum = 0
    detected_angles = []
    for lm in prev_n_landmarks_left:
        if(lm[0].visibility >=VISIBILITY_THRESHOLD and lm[1].visibility >=VISIBILITY_THRESHOLD and lm[2].visibility >=VISIBILITY_THRESHOLD):
            detected+=1
            angle = get_angle(lm[1], lm[1], lm[0], lm[2])
            detected_angles.append(angle)
            angle_sum += angle

    if(detected == 0):
        return "Not Detected"
    mean_angle = angle_sum/detected
    stddev = 0
    for angle in detected_angles:
        stddev += (angle-mean_angle)**2
    stddev/=detected
    stddev = stddev**(0.5)

    strx = ""
    if detected < len(prev_n_landmarks_left)/2 or stddev > STANDARD_DEVIATION_THRESHOLD:
        print("stddev > STANDARD_DEVIATION_THRESHOLD: ", bool(stddev > STANDARD_DEVIATION_THRESHOLD))
        strx = "Not Detected"
    else:
        strx = str(mean_angle)
    
    return strx

def get_right_elbow_angle(prev_n_landmarks_right):
    detected = 0
    angle_sum = 0
    detected_angles = []
    for lm in prev_n_landmarks_right:
        if(lm[0].visibility >=VISIBILITY_THRESHOLD and lm[1].visibility >=VISIBILITY_THRESHOLD and lm[2].visibility >=VISIBILITY_THRESHOLD):
            detected+=1
            angle = get_angle(lm[1], lm[1], lm[0], lm[2])
            detected_angles.append(angle)
            angle_sum += angle

    if(detected == 0):
        return "Not Detected"

    mean_angle = angle_sum/detected
    stddev = 0
    for angle in detected_angles:
        stddev += (angle-mean_angle)**2
    stddev/=detected
    stddev = stddev**(0.5)

    strx = ""
    if detected < len(prev_n_landmarks_right)/2 or stddev > STANDARD_DEVIATION_THRESHOLD:
        print("stddev > STANDARD_DEVIATION_THRESHOLD: ", bool(stddev > STANDARD_DEVIATION_THRESHOLD))
        strx = "Not Detected"
    else:
        strx = str(mean_angle)
    
    return strx

# For webcam input:
cap = cv2.VideoCapture(0)
prev_n_images = deque() #length = LENGTH_OF_QUEUE
img = None
keyp_prev = False
keyp_curr = False

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # # To improve performance, optionally mark the image as not writeable to
    # # pass by reference.
    # image.flags.writeable = False
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # results = pose.process(image)
    # lmks = list(results.pose_landmarks.landmark)

    prev_n_images.append(image)
    if len(prev_n_images) > LENGTH_OF_QUEUE:
        prev_n_images.popleft()
    

    # keyp_curr = is_key_pressed('r')
    # print(keyp_curr)
    if keyp_curr == True and keyp_prev == False:
        left_landmarks = []
        right_landmarks = []

        for imagex in prev_n_images:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            imagex.flags.writeable = False
            imagex = cv2.cvtColor(imagex, cv2.COLOR_BGR2RGB)
            results = pose.process(imagex)
            lmks = list(results.pose_landmarks.landmark)

            left_landmarks.append((lmks[11], lmks[13], lmks[15]))
            right_landmarks.append((lmks[12], lmks[14], lmks[16]))
        
        str1 = f"Left Elbow Angle: {get_left_elbow_angle(left_landmarks)}"
        str2 = f"Right Elbow Angle: {get_right_elbow_angle(right_landmarks)}"
        img = cv2.flip(image, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.putText(img, str1, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, str2, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    keyp_prev = keyp_curr
    # if key == ord('q'):
    #     break
    # elif key == ord('r'):
    #     print("r Key pressed")
    #     str1 = f"Left Elbow Angle: {get_left_elbow_angle(list(prev_n_landmarks_left))}"
    #     str2 = f"Right Elbow Angle: {get_right_elbow_angle(list(prev_n_landmarks_right))}"
    #     img = cv2.flip(image, 1)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img = cv2.putText(img, str1, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #     img = cv2.putText(img, str2, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #     # cv2.imshow("Angle Detected Image", img)
    #     # cv2.waitKey(1)



    # Draw the pose annotation on the image.
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    if img is not None:
        cv2.imshow("Angle Detected Image", img)
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    key = cv2.waitKey(1)
    if key == ord('r'):
        keyp_curr = True
    else:
        keyp_curr = False
        if(key == 27):
            break
cap.release()