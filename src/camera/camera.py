#! /usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

class WebCam():
    def __init__(self, webcam_id=1):
        # self.webcam = cv2.VideoCapture(webcam_id)
        # print(self.webcam)
        self.pipeline = rs.pipeline()
        self.pipeline.start()

        self.H_canvas = None

    def get_rgb_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def get_depth_image(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        return depth_colormap

    def get_canvas(self):
        if self.H_canvas is None:
            self.calibrate_canvas()
        img = self.get_rgb_image()
        canvas = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        return canvas

    def calibrate_canvas(self):
        import matplotlib.pyplot as plt
        img = self.get_rgb_image()
        h,w = img.shape[0], img.shape[1]
        plt.imshow(img)
        plt.title("Select corners of canvas. First is top-left, then clock-wise.")
        self.canvas_points = np.array(plt.ginput(n=4))
        true_points = np.array([[0,0],[w,0], [w,h],[0,h]])
        self.H_canvas, _ = cv2.findHomography(self.canvas_points, true_points)
        img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))

        plt.imshow(img1_warp)
        plt.title('Hopefully this looks like just the canvas')
        plt.show()

    def test(self):
        img = self.get_rgb_image()
        d = self.get_depth_image()

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(d)
        plt.show()

def increase_brightness(img, value=30):
    # https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img