import cv2
import mediapipe as mp
import math
import cmath
import numpy as np

# There are three approaches of finding angle among which an approach is selected using the MODE variable
# 1. MODE = 0: Pair up the finger joints in 4 pairs. For each pair join each joint with the wrist and calculate angle. Then take average of all angles
# 2. MODE = 1: Join the MCP and TIP joints of each finger respectively and then calculate the angle between the two lines. (Most effective)
# 3. MODE = 2: Join the TIP joints of both fingerswith the wrist joint and calculate the angle between them.
MODE = 1

# get_angle joins the points p0, p2 and p1, p3 and then find the angle between them in degrees. Angle less than 180 degree will be returned.
def get_angle(p0, p1, p2, p3):
  v1 = complex(p2.x-p0.x, p2.y-p0.y)
  v2 = complex(p3.x-p1.x, p3.y-p1.y)
  angle = abs(math.degrees(cmath.phase(v1))-math.degrees(cmath.phase(v2)))
  if angle>180:
      angle = 360-angle
  return angle

# get_angles find angles between each pair of adjacent fingers using the hand_landmarks
def get_angles(hand_landmarks):
  landmarks_list = list(hand_landmarks.landmark)
  wrist = landmarks_list[0] 
  thumb = [landmarks_list[2], landmarks_list[3], landmarks_list[4]]
  index = [landmarks_list[5], landmarks_list[6], landmarks_list[7], landmarks_list[8]]
  middle = [landmarks_list[9], landmarks_list[10], landmarks_list[11], landmarks_list[12]]
  ring = [landmarks_list[13], landmarks_list[14], landmarks_list[15], landmarks_list[16]]
  pinky = [landmarks_list[17], landmarks_list[18], landmarks_list[19], landmarks_list[20]]

  all_angles = []
  if MODE==0:
    angles = [] # As thumb has only 3 joints so computed seperately
    angles.append(get_angle(wrist, wrist, thumb[0], index[0]))
    angles.append(get_angle(wrist, wrist, thumb[1], index[1]))
    angles.append(get_angle(wrist, wrist, thumb[2], index[2]))
    angles.append(get_angle(wrist, wrist, thumb[-1], index[-1]))
    
    all_angles.append(np.mean(angles))

    angles = []
    for i in range(4):
      angles.append(get_angle(wrist, wrist, index[i], middle[i]))
    all_angles.append(np.mean(angles))

    angles = []
    for i in range(4):
      angles.append(get_angle(wrist, wrist, middle[i], ring[i]))
    all_angles.append(np.mean(angles))

    angles = []
    for i in range(4):
      angles.append(get_angle(wrist, wrist, ring[i], pinky[i]))
    all_angles.append(np.mean(angles))

  elif MODE==1:
    all_angles.append(get_angle(thumb[0], index[0], thumb[-1], index[-1]))
    all_angles.append(get_angle(index[0], middle[0], index[-1], middle[-1]))
    all_angles.append(get_angle(middle[0], ring[0], middle[-1], ring[-1]))
    all_angles.append(get_angle(ring[0], pinky[0], ring[-1], pinky[-1]))

  elif MODE==2:
    all_angles.append(get_angle(wrist, wrist, thumb[-1], index[-1]))
    all_angles.append(get_angle(wrist, wrist, index[-1], middle[-1]))
    all_angles.append(get_angle(wrist, wrist, middle[-1], ring[-1]))
    all_angles.append(get_angle(wrist, wrist, ring[-1], pinky[-1]))

  return all_angles





if __name__ == "__main__":

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  # For static images:
  IMAGE_FILES = []
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      print('Handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      cv2.imwrite(
          '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
      # Draw hand world landmarks.
      if not results.multi_hand_world_landmarks:
        continue
      for hand_world_landmarks in results.multi_hand_world_landmarks:
        mp_drawing.plot_landmarks(
          hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

  # For webcam input:
  cap = cv2.VideoCapture(0)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  size = (frame_width, frame_height)
  result = cv2.VideoWriter('hand.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            20, size)
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      str1 = "thumb and index: "
      str2 = "index and middle: "
      str3 = "middle and ring: "
      str4 = "ring and pinky: "

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          print("landmarks: ", list(hand_landmarks.landmark))
          print("type: ", type(list(hand_landmarks.landmark)[0]))
          print("Connections: ", mp_hands.HAND_CONNECTIONS)
          all_angles = get_angles(hand_landmarks)
          print("Angle b/w thumb and index: ", all_angles[0])
          str1 += str(int(all_angles[0]))
          print("Angle b/w index and middle: ", all_angles[1])
          str2 += str(int(all_angles[1]))
          print("Angle b/w middle and ring: ", all_angles[2])
          str3 += str(int(all_angles[2]))
          print("Angle b/w ring and pinky: ", all_angles[3])
          str4 += str(int(all_angles[3]))
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      img = cv2.flip(image, 1)
      img = cv2.putText(img, str1, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 255), 1, cv2.LINE_AA)
      img = cv2.putText(img, str2, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 255), 1, cv2.LINE_AA)
      img = cv2.putText(img, str3, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 255), 1, cv2.LINE_AA)
      img = cv2.putText(img, str4, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
      result.write(img)
      cv2.imshow('hands', img)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()