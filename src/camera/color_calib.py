#! /usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .macduff import find_macbeth, expected_colors

'''
Code to correct color of photo based on results from a Macbeth color checker.
Author: Jason Xu

Approach:
    -Find color checker in image and the average RGB values in each color checker region
    -Solve least squares equation to match the 24 regions in the image against their actual values
    -Obtain a (3, 4) transform matrix that can be applied to each pixel in an image, and save this
    -White balance image by scaling red and green channels globally according to moderately bright grey square

References:
    -Color checker finding code and reference checker values from https://github.com/mathandy/python-macduff-colorchecker-detector
    -Overall approach inspired by content from CMU 15-663 Computational Photography course
'''

# manipulates image matrix to enable matmul with transformation matrix quickly
# reshapes [h, w, 3] image array to [4, h*w]
# multiply with [3, 4] transform matrix to get [3, h*w] result image
# reshape back to original shape
def fast_tmat_mult (img, t_mat):
    x, y, z = np.shape(img)
    # reshape image to [3, h*w]
    img_flat = np.reshape(img.flatten(), (3, x * y), order='F')

    # add 1 to each pixel for homogenous coordinates
    img_homo = np.vstack((img_flat, np.ones((np.shape(img_flat))[1])))

    # multiply with [3, 4] transform matrix
    img_mult_res = np.matmul(t_mat, img_homo)

    # flatten, reorder, reshape
    img_mult_flat = img_mult_res.flatten(order='F')
    new_img = np.reshape(img_mult_flat, np.shape(img))
    
    return new_img

# corrects image based on transform matrix and white balance
# NOTE takes in images as RGB
def color_calib(in_img, tmat, greyval):
    # color calibration
    corrected = fast_tmat_mult(in_img, tmat)

    # white balancing based on 4th darkest grey square 
    b = greyval[0]
    g = greyval[1]
    r = greyval[2]

    corrected[:,:,0] *= r / b
    corrected[:,:,1] *= r / g

    # clip values and return
    final = corrected.clip(0, 255).astype(np.uint8)

    return final

# color corrects image based on given color checker values
def find_calib_params(img_path, disp_results=False):
    out, colorchecker = find_macbeth(img_path)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # get bgr version of expected values and actual checker values
    ref = expected_colors
    checkr_bgr = colorchecker.values.reshape(24, 3)

    checkr = checkr_bgr.copy()
    checkr[:,0] = checkr[:,2]
    checkr[:,2] = checkr_bgr[:,0]

    refr = ref.reshape(24, 3)

    original = out.copy()

    # set up least squares equation to solve for transform matrix
    A = np.zeros((24 * 3, 12))
    b = refr.flatten()

    for i in range(24):
        for j in range(3):
            ind = i * 3 + j
            offset = 4 * j
            A[ind, offset:offset + 4] = np.append(checkr[i], 1)
            
    t = (np.linalg.lstsq(A, b, rcond=None))[0]
    t_mat = t.reshape((3, 4), order='C')
    
    # check color-correct grey square
    greyval = checkr[-4]
    #greyval = t_mat @ np.append(greyval, 1)
    greyval = np.dot(t_mat, np.append(greyval, 1))

    # show results on original image
    if (disp_results):
        img = plt.imread(img_path)
        corrected = color_calib(img, t_mat, greyval)

        final = corrected.clip(0, 255).astype(np.uint8)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        plt.imsave('correct.jpg', final)

        out, _ = find_macbeth('correct.jpg')
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        fix, ax = plt.subplots(1,2)
        ax[0].imshow(original)
        ax[1].imshow(np.transpose(out, (1,0,2))[:,::-1,:])
        plt.title('Color Callibration results')
        for a in ax: a.set_xticks([]), a.set_yticks([])
        plt.show()

    return t_mat, greyval

if __name__ == '__main__':
    tmat, greyval = find_calib_params('test_manip.jpg')

    img = plt.imread('test_manip.jpg')
    plt.imshow(img)
    plt.show()
    plt.imshow(color_calib(img, tmat, greyval))
    plt.show()

