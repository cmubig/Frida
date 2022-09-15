from dslr_gphoto import *

cam = camera_init()

path, img = capture_image(cam)
path, img = capture_image(cam)
path, img = capture_image(cam)

import matplotlib.pyplot as plt
plt.imshow(img)
#plt.show()
