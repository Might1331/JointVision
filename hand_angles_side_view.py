import cv2
import mediapipe as mp
import math
import cmath
import numpy as np


def get_angle(p0, p1, p2, p3):
  v1 = complex(p2.x-p0.x, p2.y-p0.y)
  v2 = complex(p3.x-p1.x, p3.y-p1.y)
  angle = abs(math.degrees(cmath.phase(v1))-math.degrees(cmath.phase(v2)))
  if angle>180:
      angle = 360-angle
  return angle

def get_angles(hand_landmarks):
  landmarks_list = list(hand_landmarks.landmark)
  wrist = landmarks_list[0]
  thumb = [landmarks_list[2], landmarks_list[3], landmarks_list[4]]
  index = [landmarks_list[5], landmarks_list[6], landmarks_list[7], landmarks_list[8]]
  middle = [landmarks_list[9], landmarks_list[10], landmarks_list[11], landmarks_list[12]]
  ring = [landmarks_list[13], landmarks_list[14], landmarks_list[15], landmarks_list[16]]
  pinky = [landmarks_list[17], landmarks_list[18], landmarks_list[19], landmarks_list[20]]

  top_tip_finger = []
  bottom_tip_finger = []
  min_y = np.inf
  max_y = -np.inf

  for finger in [thumb, index, middle, ring, pinky]:
      if finger[-1].y<min_y:
          top_tip_finger = finger
          min_y = finger[-1].y

  angle = max(get_angle(top_tip_finger[0], wrist, top_tip_finger[-1], index[-1]), get_angle(top_tip_finger[0], wrist, top_tip_finger[-1], middle[-1]))

  return angle





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
  result = cv2.VideoWriter('flat_hand.avi', 
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
      strf = "Angle b/w lifted finger and palm: "

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          print("landmarks: ", list(hand_landmarks.landmark))
          print("type: ", type(list(hand_landmarks.landmark)[0]))
          print("Connections: ", mp_hands.HAND_CONNECTIONS)
          lift_angle = get_angles(hand_landmarks)
          print("Angle b/w lifted finger and palm: ", lift_angle)
          strf += str(int(lift_angle))+" "
          
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      img = cv2.flip(image, 1)
      img = cv2.putText(img, strf, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
      result.write(img)
      cv2.imshow('Lifted Finger angle detection', img)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()