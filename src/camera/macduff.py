#!/usr/bin/env python
"""Python-Macduff: "the Macbeth ColorChecker finder", ported to Python.

Original C++ code: github.com/ryanfb/macduff/

Usage:
    # if pixel-width of color patches is unknown,
    $ python macduff.py examples/test.jpg result.png > result.csv

    # if pixel-width of color patches is known to be, e.g. 65,
    $ python macduff.py examples/test.jpg result.png 65 > result.csv
"""
from __future__ import print_function, division
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from math import sqrt
from sys import stderr, argv
from copy import copy
import os


_root = os.path.dirname(os.path.realpath(__file__))


# Each color square must takes up more than this percentage of the image
MIN_RELATIVE_SQUARE_SIZE = 0.0001

DEBUG = False

MACBETH_WIDTH = 6
MACBETH_HEIGHT = 4
MACBETH_SQUARES = MACBETH_WIDTH * MACBETH_HEIGHT

MAX_CONTOUR_APPROX = 50  # default was 7


# pick the colorchecker values to use -- several options available in
# the `color_data` subdirectory
# Note: all options are explained in detail at
# http://www.babelcolor.com/colorchecker-2.htm
color_data = os.path.join(_root, 'color_data',
                          'xrite_post-2014.csv')
expected_colors = np.flip(np.loadtxt(color_data, delimiter=','), 1)
expected_colors = expected_colors.reshape(MACBETH_HEIGHT, MACBETH_WIDTH, 3)


# a class to simplify the translation from c++
class Box2D:
    """
    Note: The Python equivalent of `RotatedRect` and `Box2D` objects 
    are tuples, `((center_x, center_y), (w, h), rotation)`.
    Example:
    >>> cv.boxPoints(((0, 0), (2, 1), 0))
    array([[-1. ,  0.5],
           [-1. , -0.5],
           [ 1. , -0.5],
           [ 1. ,  0.5]], dtype=float32)
    >>> cv.boxPoints(((0, 0), (2, 1), 90))
    array([[-0.5, -1. ],
           [ 0.5, -1. ],
           [ 0.5,  1. ],
           [-0.5,  1. ]], dtype=float32)
    """
    def __init__(self, center=None, size=None, angle=0, rrect=None):
        if rrect is not None:
            center, size, angle = rrect

        # self.center = Point2D(*center)
        # self.size = Size(*size)
        self.center = center
        self.size = size
        self.angle = angle  # in degrees

    def rrect(self):
        return self.center, self.size, self.angle


def crop_patch(center, size, image):
    """Returns mean color in intersection of `image` and `rectangle`."""
    x, y = center - np.array(size)/2
    w, h = size
    x0, y0, x1, y1 = map(round, [x, y, x + w, y + h])
    return image[int(max(y0, 0)): int(min(y1, image.shape[0])),
                 int(max(x0, 0)): int(min(x1, image.shape[1]))]


def contour_average(contour, image):
    """Assuming `contour` is a polygon, returns the mean color inside it.

    Note: This function is inefficiently implemented!!! 
    Maybe using drawing/fill functions would improve speed.
    """

    # find up-right bounding box
    xbb, ybb, wbb, hbb = cv.boundingRect(contour)

    # now found which points in bounding box are inside contour and sum
    def is_inside_contour(pt):
        return cv.pointPolygonTest(contour, pt, False) > 0

    from itertools import product as catesian_product
    from operator import add
    from functools import reduce
    bb = catesian_product(range(max(xbb, 0), min(xbb + wbb,  image.shape[1])),
                          range(max(ybb, 0), min(ybb + hbb,  image.shape[0])))
    pts_inside_of_contour = [xy for xy in bb if is_inside_contour(xy)]

    # pts_inside_of_contour = list(filter(is_inside_contour, bb))
    color_sum = reduce(add, (image[y, x] for x, y in pts_inside_of_contour))
    return color_sum / len(pts_inside_of_contour)


def rotate_box(box_corners):
    """NumPy equivalent of `[arr[i-1] for i in range(len(arr)]`"""
    return np.roll(box_corners, 1, 0)


def check_colorchecker(values, expected_values=expected_colors):
    """Find deviation of colorchecker `values` from expected values."""
    diff = (values - expected_values).ravel(order='K')
    return sqrt(np.dot(diff, diff))


# def check_colorchecker_lab(values):
#     """Converts to Lab color space then takes Euclidean distance."""
#     lab_values = cv.cvtColor(values, cv.COLOR_BGR2Lab)
#     lab_expected = cv.cvtColor(expected_colors, cv.COLOR_BGR2Lab)
#     return check_colorchecker(lab_values, lab_expected)


def draw_colorchecker(colors, centers, image, radius):
    for observed_color, expected_color, pt in zip(colors.reshape(-1, 3),
                                                  expected_colors.reshape(-1, 3),
                                                  centers.reshape(-1, 2)):
        x, y = pt.astype(int)
        cv.circle(image, (x, y), radius//2, expected_color.tolist(), -1)
        cv.circle(image, (x, y), radius//4, observed_color.tolist(), -1)
    return image


class ColorChecker:
    def __init__(self, error, values, points, size):
        self.error = error
        self.values = values
        self.points = points
        self.size = size


def find_colorchecker(boxes, image, debug_filename=None, use_patch_std=True,
                      debug=DEBUG):

    points = np.array([[box.center[0], box.center[1]] for box in boxes])
    passport_box = cv.minAreaRect(points.astype('float32'))
    (x, y), (w, h), a = passport_box
    box_corners = cv.boxPoints(passport_box)

    # sort `box_corners` to be in order tl, tr, br, bl
    top_corners = sorted(enumerate(box_corners), key=lambda c: c[1][1])[:2]
    top_left_idx = min(top_corners, key=lambda c: c[1][0])[0]
    box_corners = np.roll(box_corners, -top_left_idx, 0)
    tl, tr, br, bl = box_corners

    if debug:
        debug_images = [copy(image), copy(image)]
        for box in boxes:
            pts_ = [cv.boxPoints(box.rrect()).astype(np.int32)]
            cv.polylines(debug_images[0], pts_, True, (255, 0, 0))
        pts_ = [box_corners.astype(np.int32)]
        cv.polylines(debug_images[0], pts_, True, (0, 0, 255))

        bgrp = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for pt, c in zip(box_corners, bgrp):
            cv.circle(debug_images[0], tuple(np.array(pt, dtype='int')), 10, c)
        # cv.imwrite(debug_filename, np.vstack(debug_images))

        print("Box:\n\tCenter: %f,%f\n\tSize: %f,%f\n\tAngle: %f\n" 
              "" % (x, y, w, h, a), file=stderr)

    landscape_orientation = True  # `passport_box` is wider than tall
    if norm(tr - tl) < norm(bl - tl):
        landscape_orientation = False

    average_size = int(sum(min(box.size) for box in boxes) / len(boxes))
    if landscape_orientation:
        dx = (tr - tl)/(MACBETH_WIDTH - 1)
        dy = (bl - tl)/(MACBETH_HEIGHT - 1)
    else:
        dx = (bl - tl)/(MACBETH_WIDTH - 1)
        dy = (tr - tl)/(MACBETH_HEIGHT - 1)

    # calculate the averages for our oriented colorchecker
    checker_dims = (MACBETH_HEIGHT, MACBETH_WIDTH)
    patch_values = np.empty(checker_dims + (3,), dtype='float32')
    patch_points = np.empty(checker_dims + (2,), dtype='float32')
    sum_of_patch_stds = np.array((0.0, 0.0, 0.0))
    for x in range(MACBETH_WIDTH):
        for y in range(MACBETH_HEIGHT):
            center = tl + x*dx + y*dy

            px, py = center
            img_patch = crop_patch(center, [average_size]*2, image)

            if not landscape_orientation:
                y = MACBETH_HEIGHT - 1 - y

            patch_points[y, x] = center
            patch_values[y, x] = img_patch.mean(axis=(0, 1))
            sum_of_patch_stds += img_patch.std(axis=(0, 1))

            if debug:
                rect = (px, py), (average_size, average_size), 0
                pts_ = [cv.boxPoints(rect).astype(np.int32)]
                cv.polylines(debug_images[1], pts_, True, (0, 255, 0))
    if debug:
        cv.imwrite(debug_filename, np.vstack(debug_images))

    # determine which orientation has lower error
    orient_1_error = check_colorchecker(patch_values)
    orient_2_error = check_colorchecker(patch_values[::-1, ::-1])

    if orient_1_error > orient_2_error:  # rotate by 180 degrees
        patch_values = patch_values[::-1, ::-1]
        patch_points = patch_points[::-1, ::-1]

    if use_patch_std:
        error = sum_of_patch_stds.mean() / MACBETH_SQUARES
    else:
        error = min(orient_1_error, orient_2_error)

    if debug:
        print("dx =", dx, file=stderr)
        print("dy =", dy, file=stderr)
        print("Average contained rect size is %d\n" % average_size, file=stderr)
        print("Orientation 1: %f\n" % orient_1_error, file=stderr)
        print("Orientation 2: %f\n" % orient_2_error, file=stderr)
        print("Error: %f\n" % error, file=stderr)

    return ColorChecker(error=error,
                        values=patch_values,
                        points=patch_points,
                        size=average_size)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


# https://github.com/opencv/opencv/blob/master/samples/python/squares.py
# Note: This is similar to find_quads, added to hastily add support to
# the `patch_size` parameter
def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)

            tmp = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            try:
                contours, _ = tmp
            except ValueError:  # OpenCV version < 4.0.0
                bin, contours, _ = tmp

            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if (len(cnt) == 4 and cv.contourArea(cnt) > 1000
                        and cv.isContourConvex(cnt)):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i + 2) % 4])
                                   for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def is_right_size(quad, patch_size, rtol=.25):
    """Determines if a (4-point) contour is approximately the right size."""
    cw = abs(np.linalg.norm(quad[0] - quad[1]) - patch_size) < rtol*patch_size
    ch = abs(np.linalg.norm(quad[0] - quad[3]) - patch_size) < rtol*patch_size
    return cw and ch


# stolen from icvGenerateQuads
def find_quad(src_contour, min_size, debug_image=None):

    for max_error in range(2, MAX_CONTOUR_APPROX + 1):
        dst_contour = cv.approxPolyDP(src_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

        # we call this again on its own output, because sometimes
        # cvApproxPoly() does not simplify as much as it should.
        dst_contour = cv.approxPolyDP(dst_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

    # reject non-quadrangles
    is_acceptable_quad = False
    is_quad = False
    if len(dst_contour) == 4 and cv.isContourConvex(dst_contour):
        is_quad = True
        perimeter = cv.arcLength(dst_contour, closed=True)
        area = cv.contourArea(dst_contour, oriented=False)

        d1 = np.linalg.norm(dst_contour[0] - dst_contour[2])
        d2 = np.linalg.norm(dst_contour[1] - dst_contour[3])
        d3 = np.linalg.norm(dst_contour[0] - dst_contour[1])
        d4 = np.linalg.norm(dst_contour[1] - dst_contour[2])

        # philipg.  Only accept those quadrangles which are more square
        # than rectangular and which are big enough
        cond = (d3/1.1 < d4 < d3*1.1 and
                d3*d4/1.5 < area and
                min_size < area and
                d1 >= 0.15*perimeter and
                d2 >= 0.15*perimeter)

        if not cv.CALIB_CB_FILTER_QUADS or area > min_size and cond:
            is_acceptable_quad = True
            # return dst_contour
    if debug_image is not None:
        cv.drawContours(debug_image, [src_contour], -1, (255, 0, 0), 1)
        if is_acceptable_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 255, 0), 1)
        elif is_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 0, 255), 1)
        return debug_image

    if is_acceptable_quad:
        return dst_contour
    return None


def find_macbeth(img, patch_size=None, is_passport=False, debug=DEBUG,
                 min_relative_square_size=MIN_RELATIVE_SQUARE_SIZE):
    macbeth_img = img
    if isinstance(img, str):
        macbeth_img = cv.imread(img)
    macbeth_original = copy(macbeth_img)
    macbeth_split = cv.split(macbeth_img)

    # threshold each channel and OR results together
    block_size = int(min(macbeth_img.shape[:2]) * 0.02) | 1
    macbeth_split_thresh = []
    for channel in macbeth_split:
        res = cv.adaptiveThreshold(channel,
                                    255,
                                    cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY_INV,
                                    block_size,
                                    C=6)
        macbeth_split_thresh.append(res)
    adaptive = np.bitwise_or(*macbeth_split_thresh)

    if debug:
        print("Used %d as block size\n" % block_size, file=stderr)
        cv.imwrite('debug_threshold.png',
                    np.vstack(macbeth_split_thresh + [adaptive]))

    # do an opening on the threshold image
    element_size = int(2 + block_size / 10)
    shape, ksize = cv.MORPH_RECT, (element_size, element_size)
    element = cv.getStructuringElement(shape, ksize)
    adaptive = cv.morphologyEx(adaptive, cv.MORPH_OPEN, element)

    if debug:
        print("Used %d as element size\n" % element_size, file=stderr)
        cv.imwrite('debug_adaptive-open.png', adaptive)

    # find contours in the threshold image
    tmp = cv.findContours(image=adaptive,
                          mode=cv.RETR_LIST,
                          method=cv.CHAIN_APPROX_SIMPLE)
    try:
        contours, _ = tmp
    except ValueError:  # OpenCV < 4.0.0
        adaptive, contours, _ = tmp

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_all_contours.png', show_contours)

    min_size = np.product(macbeth_img.shape[:2]) * min_relative_square_size

    def is_seq_hole(c):
        return cv.contourArea(c, oriented=True) > 0

    def is_big_enough(contour):
        _, (w, h), _ = cv.minAreaRect(contour)
        return w * h >= min_size

    # filter out contours that are too small or clockwise
    contours = [c for c in contours if is_big_enough(c) and is_seq_hole(c)]

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_big_contours.png', show_contours)

        debug_img = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        for c in contours:
            debug_img = find_quad(c, min_size, debug_image=debug_img)
        cv.imwrite("debug_quads.png", debug_img)

    if contours:
        if patch_size is None:
            initial_quads = [find_quad(c, min_size) for c in contours]
        else:
            initial_quads = [s for s in find_squares(macbeth_original)
                             if is_right_size(s, patch_size)]
            if is_passport and len(initial_quads) <= MACBETH_SQUARES:
                qs = [find_quad(c, min_size) for c in contours]
                qs = [x for x in qs if x is not None]
                initial_quads = [x for x in qs if is_right_size(x, patch_size)]
        initial_quads = [q for q in initial_quads if q is not None]
        initial_boxes = [Box2D(rrect=cv.minAreaRect(q)) for q in initial_quads]

        if debug:
            show_quads = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
            cv.drawContours(show_quads, initial_quads, -1, (0, 255, 0))
            cv.imwrite('debug_quads2.png', show_quads)
            print("%d initial quads found", len(initial_quads), file=stderr)

        if is_passport or (len(initial_quads) > MACBETH_SQUARES):
            if debug:
                print(" (probably a Passport)\n", file=stderr)

            # set up the points sequence for cvKMeans2, using the box centers
            points = np.array([box.center for box in initial_boxes],
                              dtype='float32')

            # partition into two clusters: passport and colorchecker
            criteria = \
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, clusters, centers = \
                cv.kmeans(data=points,
                           K=2,
                           bestLabels=None,
                           criteria=criteria,
                           attempts=100,
                           flags=cv.KMEANS_RANDOM_CENTERS)

            partitioned_quads = [[], []]
            partitioned_boxes = [[], []]
            for i, cluster in enumerate(clusters.ravel()):
                partitioned_quads[cluster].append(initial_quads[i])
                partitioned_boxes[cluster].append(initial_boxes[i])

            debug_fns = [None, None]
            if debug:
                debug_fns = ['debug_passport_box_%s.jpg' % i for i in (0, 1)]

                # show clustering
                img_clusters = []
                for cl in partitioned_quads:
                    img_copy = copy(macbeth_original)
                    cv.drawContours(img_copy, cl, -1, (255, 0, 0))
                    img_clusters.append(img_copy)
                cv.imwrite('debug_clusters.jpg', np.vstack(img_clusters))

            # check each of the two partitioned sets for the best colorchecker
            partitioned_checkers = []
            for cluster_boxes, fn in zip(partitioned_boxes, debug_fns):
                partitioned_checkers.append(
                    find_colorchecker(cluster_boxes, macbeth_original, fn,
                                      debug=debug))

            # use the colorchecker with the lowest error
            found_colorchecker = min(partitioned_checkers,
                                     key=lambda checker: checker.error)

        else:  # just one colorchecker to test
            debug_img = None
            if debug:
                debug_img = "debug_passport_box.jpg"
                print("\n", file=stderr)

            found_colorchecker = \
                find_colorchecker(initial_boxes, macbeth_original, debug_img,
                                  debug=debug)

        # render the found colorchecker
        draw_colorchecker(found_colorchecker.values,
                          found_colorchecker.points,
                          macbeth_img,
                          found_colorchecker.size)

        # print out the colorchecker info
        for color, pt in zip(found_colorchecker.values.reshape(-1, 3),
                             found_colorchecker.points.reshape(-1, 2)):
            b, g, r = color
            x, y = pt
            if debug:
                print("%.0f,%.0f,%.0f,%.0f,%.0f\n" % (x, y, r, g, b))
        if debug:
            print("%0.f\n%f\n" 
                  "" % (found_colorchecker.size, found_colorchecker.error))
    else:
        raise Exception('Something went wrong -- no contours found')
    return macbeth_img, found_colorchecker


def write_results(colorchecker, filename=None):
    mes = ',r,g,b\n'
    for k, (b, g, r) in enumerate(colorchecker.values.reshape(1, 3)):
        mes += '{},{},{},{}\n'.format(k, r, g, b)

    if filename is None:
        print(mes)
    else:
        with open(filename, 'w+') as f:
            f.write(mes)


if __name__ == '__main__':
    if len(argv) == 3:
        out, colorchecker = find_macbeth(argv[1])
        cv.imwrite(argv[2], out)
    elif len(argv) == 4:
        out, colorchecker = find_macbeth(argv[1], patch_size=float(argv[3]))
        cv.imwrite(argv[2], out)
    else:
        print('Usage: %s <input_image> <output_image> <(optional) patch_size>\n'
              '' % argv[0], file=stderr)
    print(colorchecker.values.shape)
