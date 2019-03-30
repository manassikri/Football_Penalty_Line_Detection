import cv2
import numpy as np
 
def nothing(x):
    pass
 
cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
 
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
 
while True:
    frame = cv2.imread("D:/Python/football_dataset/football8.jpg")
    #print(np.mean(frame[1]))
    l=np.mean(frame)
    l=int(l)
    #print("l: ",l)
    s=int(np.mean(frame[1]))
    #print("s: ",s)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #print(frame)
    #print("1: ",np.mean(frame[0]))
    #print("2: ",np.mean(frame[1]))
    #print("3: ",np.mean(frame[2]))
    p=np.mean(frame[1])/6
    p=int(p)
    #print(p)
    
 
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
    #lower = np.array([l_h, l_s, l_v])
    #upper = np.array([u_h, u_s,u_v])
    if (255-s)>190:
        t=255-s
    else:
        t=195
    #print("t: ",t)
    
    if (2*l)>202:
        v=202
    elif (2*l)<180:
        v=180

    else:
        v=2*l+5
    lower = np.array([30, p, v])
    upper = np.array([179-l,t,255])
    mask = cv2.inRange(hsv, lower, upper)
 
    result = cv2.bitwise_and(frame, frame, mask=mask)
 
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    gray=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("blur",blur)

    canny = cv2.Canny(blur, 50, 150)
    cv2.imshow("canny",canny)
    kernelOpen=np.ones((1,1))
    kernelClose=np.ones((3,3))

    maskOpen=cv2.morphologyEx(canny,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)

    rho = 1

    # 1 degree
    theta = (np.pi/180) * 1
    threshold = 10
    min_line_length = 30
    max_line_gap = 10

    lines=cv2.HoughLinesP(maskClose, rho, theta, threshold, np.array([]),
                         minLineLength=min_line_length, maxLineGap=max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), [0,255,0], 2)
            m=(y2-y1)/(x2-x1)
            #print ("y="+str(round(m,2))+ "x+"+str(y2-y1))
                
    cv2.imshow('output',frame)

    
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
