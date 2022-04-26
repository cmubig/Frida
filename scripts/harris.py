import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


NUM_CORNERS = 4
corners = []


def search_corner(harris_dest, coord, search_size, show_search=False):
    x, y = coord
    # crop regions in search box for Harris processed image
    sbox_harris = harris_dest[max(0, y-search_size):y+search_size, max(0,x-search_size):x+search_size]
   
    # find the location of max probability in the Harris image
    ymax, xmax = np.unravel_index(sbox_harris.argmax(), sbox_harris.shape)
    
    # show corner probabilities in search box to diagnose box size
    # if(show_search):
    #     print((xmax, ymax))
    #     plt.imshow(sbox_harris)
    #     plt.show()

    # find the location of max probability in the original image
    max_orig = [sum(x) for x in zip((xmax, ymax), (max(0,x-search_size), max(0,y-search_size)))]

    return max_orig

def find_corners(img, search_size=10, show_search=False):
    
    # show image and take input
    plt.imshow(img)
    plt.title("Select corners of canvas. First is top-left, then clock-wise.")
    points = np.array(plt.ginput(n=NUM_CORNERS)).astype(np.int64)

    # print(points)

    # convert to grayscale and calculate corner probabilities
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    harris_probs = cv.cornerHarris(gray, 2, 3, 0.04)

    # find true corner for each clicked corner
    for corner_num in range(NUM_CORNERS):
        coord = points[corner_num]
        try:
            actual_corner = np.array(search_corner(harris_probs, coord, search_size, show_search))
        except:
            print('Error finding real corner')
            actual_corner = coord
        points[corner_num] = actual_corner

    # print(points)
    return points

        

    