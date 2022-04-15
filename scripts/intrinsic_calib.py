# Camera Intrinsic Calibration 
# Author: Jason Xu

import numpy as np
import cv2
from skimage import io

# Compute camera intrinsics matrix
# Code from 15-463 Computational Photography at Carnegie Mellon

#images: list of filenames for checkboard calibration files
#checkerboard: dimension of the inner corners
#dW: corner refinement window size. should be smaller for lower resolution images
def computeIntrinsic(images, checkerboard, dW):    
    # Defining the dimensions of checkerboard
    # Number of inner corners, hard coded to class example
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    img_shape = None

    # Extracting path of individual image stored in a given directory
    print('Displaying chessboard corners. Press any button to continue to next example')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            #print(corners)
            corners2 = cv2.cornerSubPix(gray, corners, dW, (-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        else:
            print("error: checkerboard not found")
        
        io.imshow(img)
        io.show()

    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
        ret:
        mtx: camera matrix
        dist: distortion coefficients
        
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("Camera matrix: \n")
    print(mtx)
    print("Distortion: \n")
    print(dist)

    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 0, img_shape)

    return mtx, dist, newCameraMtx, roi
    

