import cv2
import sys
import os

cascPath = os.environ['CONDA_PREFIX']+ "/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml"  
print(cascPath)
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

img_path = os.environ['HOME'] +  "imgs"
alpha = 0.3
x,y,w,h = -1,-1,-1,-1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_ = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for x_,y_,w_,h_ in faces:
        if x == -1:
            x,y,w,h = faces[0]
        else:
            x,y,w,h = (1-alpha)*x+x_*alpha, (1-alpha)*y+y_*alpha, (1-alpha)*w+w_*alpha,  (1-alpha)*h+h_*alpha
        # Draw a rectangle around the faces
        cv2.rectangle(frame_, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        break
    # Display the resulting frame
    cv2.imshow('Video', frame_)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        w_offset, h_offset = 100,100
        crop = frame[int(max(x-w_offset,0)): int(min(x+w+w_offset,frame.shape[0])),int(max(y-h_offset,0)): int(min(y+h+h_offset,frame.shape[1]))]
        cv2.imshow("portrait", crop)
        #save image
        cv2.imwrite(img_path  + "portrait.png" , crop)
    if key == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
