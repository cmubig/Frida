import cv2
import sys
import os
from datetime import datetime
import easygui
import unicodedata
import numpy as np
class WebcamInterface():
    def __init__(self, num_iter=1000,num_strokes=8, cascade_path="haarcascade_frontalface_default.xml", img_path=None, alpha=0.9, ratio=None):
        self.cascPath = cascade_path if cascade_path is not None else os.environ['CONDA_PREFIX']+ "/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.img_path = os.environ['HOME'] +  "/imgs/" if img_path is None else img_path
        # create path if it does not exist
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        self.alpha = alpha
        self.x,self.y,self.w,self.h = -1,-1,-1,-1
        self.w_offset, self.h_offset = 60,125
        self.last_portrait_path = self.img_path + "last_portrait.png"
        self.sqrt_num_strokes = int(num_strokes**0.5)

        self.detect_face = True
        self.num_iter = num_iter
        self.ratio = ratio #w/h
        self.selecting = False
        self.click_x, self.click_y = 0,0
        self.camera_stream = True
        self.frame_ = None
        
    def run(self) -> bool:
        print("----------- starting webacam interface -----------")
        self.video_capture = cv2.VideoCapture(0)

        intructions = """PRESS:
'p' to take a new portrait of the selected red crop
'o' to upload an image
'c' to return to the camera stream
'm' to enable/disable manual crop
's' to finish and start painting
'q' to quit"""
        print(intructions)

        errors = 0
        def nothing(x):
            pass
        def nothing2(x):
            pass
        def select_corners(event, x,y,flags,param):
            if not self.detect_face:
                if event ==cv2.EVENT_LBUTTONDOWN:
                    self.click_x = x
                    self.click_y = y
                    self.selecting = True
                elif event == cv2.EVENT_LBUTTONUP:
                    self.selecting = False
                else:
                    if self.selecting:
                        self.h = abs(y-self.click_y)
                        self.w = abs(x-self.click_x) if self.ratio is None else self.ratio *self.h
                        self.x = self.click_x- self.w/2# + self.w_offset*self.w_offset_ratio #- self.w/2
                        self.y = self.click_y- self.h/2# + self.h_offset*self.h_offset_ratio #- self.h/2
                if self.frame_ is not None:
                    cv2.circle(self.frame_, (x,y), 5, (0,255,0),-1)
                    cv2.putText(self.frame_,'crop center',(x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                
            # print("corner_select")

        cv2.namedWindow('Video')
        

        cv2.createTrackbar('num_iter','Video',0,2000,nothing2)
        cv2.createTrackbar('sqrt_num_strokes','Video',2,15,nothing2)

        # set ticks

        cv2.createTrackbar('w_offset','Video',0,255,nothing) if self.ratio is None else None
        cv2.createTrackbar('h_offset','Video',0,255,nothing)

        cv2.createTrackbar('w_offset_ratio','Video',0,100,nothing)
        cv2.createTrackbar('h_offset_ratio','Video',0,100,nothing)



        # set initial values
        cv2.setTrackbarPos('w_offset','Video',self.w_offset) if self.ratio is None else None
        cv2.setTrackbarPos('h_offset','Video',self.h_offset)
        cv2.setTrackbarPos('w_offset_ratio','Video',50)
        cv2.setTrackbarPos('h_offset_ratio','Video',50)
        cv2.setTrackbarPos('num_iter','Video',self.num_iter)
        cv2.setTrackbarPos('sqrt_num_strokes','Video',self.sqrt_num_strokes)

        self.get_trackbar_data()
        try:
            cv2.setMouseCallback("Video", select_corners)
        except Exception as e:
            print(e)

        frame = None
        self.frame_ = None
        while True:
            # Capture frame-by-frame
            if self.camera_stream:
                try:
                    ret, frame = self.video_capture.read()
                except Exception as e:
                    ret = True
                    frame = np.zeros((500,500,3), dtype=np.uint8) if frame is None else frame
                    print("error opening camera")
                    self.video_capture = cv2.VideoCapture(0)
            else:
                ret = True
                frame = np.zeros((500,500,3), dtype=np.uint8) if frame is None else frame

            if ret:
                self.frame_ = frame.copy()
            else:
                print("no webcam connected")
                frame = np.zeros((500,500,3), dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
            except:
                faces = []

            self.get_trackbar_data()
            
            if self.detect_face:
                for x_,y_,w_,h_ in faces:
                    if self.x == -1:
                        self.x,self.y,self.w,self.h = faces[0]
                    else:
                        self.x,self.y,self.w,self.h = (1-self.alpha)*self.x+x_*self.alpha, (1-self.alpha)*self.y+y_*self.alpha, (1-self.alpha)*self.w+w_*self.alpha,  (1-self.alpha)*self.h+h_*self.alpha
                    # Draw a rectangle around the faces
                    if not self.ratio is None:
                        self.w = self.h * self.ratio
                    cv2.rectangle(self.frame_, (int(self.x), int(self.y)), (int(self.x+self.w), int(self.y+self.h)), (0, 255, 0), 2)
                    break
                cv2.putText(self.frame_, "detecting faces: ON", (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                    
            else:
                pass
                cv2.putText(self.frame_, "detecting faces: OFF", (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


            p_x_min = int(max(self.x-self.w_offset*self.w_offset_ratio,0))
            p_x_max = int(min(self.x+self.w+self.w_offset*(1-self.w_offset_ratio),frame.shape[1]))
            p_y_min = int(max(self.y-self.h_offset*self.h_offset_ratio,0))
            p_y_max = int(min(self.y+self.h+self.h_offset*(1-self.h_offset_ratio),frame.shape[0]))

            cv2.rectangle(self.frame_, (p_x_min, p_y_min), (p_x_max, p_y_max), (0, 0, 255 ), 2)


            # When everything is done, release the capture
            
            # add instructions to the frame
            for j, inst_line in enumerate(intructions.split('\n')):
                
                cv2.putText(self.frame_, inst_line, (10, int(30+j*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow('Video', self.frame_)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                
                crop = frame[p_y_min: p_y_max, p_x_min:p_x_max, :]
                self.save_last(crop)
            if key == ord('o'):
                #open image
                try:
                    uni_code = easygui.fileopenbox(msg="chose a image file " ,default="/home/big/imgs/")
                    img_path = unicodedata.normalize('NFKD', uni_code).encode('ascii','ignore')
                    crop = cv2.imread(uni_code)
                    cv2.imshow("portrait", crop)
                    #save image
                    cv2.imwrite(self.last_portrait_path , crop)

                    self.camera_stream = False

                    frame = crop
                except Exception as e:
                    print(f"could not load the file: {uni_code}")
                    print(e)
            if key == ord('c'):
                self.camera_stream = True

            if key == ord('q'):
                errors = 1
                break
            if key == ord('m'): # manual crop
                self.select_crop(frame)
            if key == ord('s'):
                break
        self.video_capture.release()

        cv2.destroyAllWindows()
            
        return errors, self.num_iter, self.num_strokes
    def get_trackbar_data(self):
        self.h_offset = cv2.getTrackbarPos('h_offset','Video') 
        self.w_offset = cv2.getTrackbarPos('w_offset','Video') if self.ratio is None else self.h_offset * self.ratio
        self.h_offset_ratio = cv2.getTrackbarPos('h_offset_ratio','Video')/100
        self.w_offset_ratio = cv2.getTrackbarPos('w_offset_ratio','Video')/100
        self.num_iter = cv2.getTrackbarPos('num_iter','Video')
        self.num_strokes = cv2.getTrackbarPos('sqrt_num_strokes','Video')**2

    def save_last(self, crop):
        cv2.imshow("portrait", crop)
        #save image
        cv2.imwrite(self.last_portrait_path , crop)
        stamp = str(datetime.now()).replace(" ","_")
        img_bk_file = self.img_path  + f"{stamp}.png"
        print(img_bk_file)
        cv2.imwrite(img_bk_file , crop)
        print("portrait saved, press 's' to execute the pipeline, or 'q' to cancel.")

    def select_crop(self, img):
        # open an image and let the user select 2 points to select the crop
        self.detect_face = not self.detect_face
        pass

    def __del__(self):
        # When everything is done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam = WebcamInterface()
    webcam.run()
    del webcam