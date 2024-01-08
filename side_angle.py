import cv2  
import numpy as np  
import math,cmath
def nothing(x):
    return

def get_angle(p0, p1, p2, p3):
    v1 = complex(p2 - p0, p3 - p1)
    v2 = complex(p2-p0, 0 - 0)
    angle = abs(math.degrees(cmath.phase(v1)) - math.degrees(cmath.phase(v2)))
    if angle > 180:
        angle = 360 - angle
    return angle

cv2.namedWindow('image')
cv2.createTrackbar('H_lower','image',0,255,nothing) 
cv2.createTrackbar('S_lower','image',0,255,nothing)
cv2.createTrackbar('V_lower','image',0,255,nothing)
cv2.createTrackbar('H_upper','image',255,255,nothing) 
cv2.createTrackbar('S_upper','image',190,255,nothing)
cv2.createTrackbar('V_upper','image',255,255,nothing)
frame=cv2.imread("Finger-Angle-Detection\image1.jpg",cv2.IMREAD_COLOR)
frame =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

h_lower=cv2.getTrackbarPos('H_lower','image')
s_lower=cv2.getTrackbarPos('S_lower','image')
v_lower=cv2.getTrackbarPos('V_lower','image')
h_upper=cv2.getTrackbarPos('H_upper','image')
s_upper=cv2.getTrackbarPos('S_upper','image')
v_upper=cv2.getTrackbarPos('V_upper','image')
lower_bound=np.array([h_lower,s_lower,v_lower])
upper_bound=np.array([h_upper,s_upper,v_upper])

mask=cv2.inRange(frame,lower_bound,upper_bound)
first_one=0
second_one=0
first_two=0
second_two=0
pos=False
print(len(mask),len(mask[0]))
# for i in range(int(len(mask[0])/6)):
for j in range(int(len(mask))):
    if pos==False and mask[j][int(len(mask[0])/6)]!=0:
        first_one=j
        pos=True
    if pos==True and mask[j][int(len(mask[0])/6)]==0:
        second_one=j
        break
pos=False
# for i in range(int(len(mask[0])/3)):
for j in range(len(mask)):
    if pos==False and mask[j][int(len(mask[0])/3)]!=0:
        first_two=j
        pos=True
    if pos==True and mask[j][int(len(mask[0])/3)]==0:
        second_two=j
        break


one_mid=(first_one+second_one)/2
two_mid=(first_two+second_two)/2
angle=get_angle(len(mask[0])/6,one_mid,len(mask[0])/3,two_mid)
print("Angle between finger and palm: ", angle)
cv2.imshow('mask',mask)  
cv2.waitKey(0)

cv2.destroyAllWindows()
  
