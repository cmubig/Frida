import pickle
import numpy as np
import cv2
from scipy import ndimage
import os
import gzip

# strokes = pickle.load(open(("cache/strokes.pkl"),'rb'))
# strokes = [cv2.resize(s,(256,128)) for s in strokes]

# strokes_processed = []
# for s in strokes:
#     t = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.uint8)
#     t[:,:,3] = (s*255.).astype(np.uint8)
#     t[:,:,2] = 255

#     # plt.imshow(t)
#     # plt.show()
#     strokes_processed.append(t)
# strokes_processed = np.array(strokes_processed)
# print(strokes_processed.shape)
# np.save(open('strokes_numpy.npy', 'wb'), strokes_processed)


def center_stroke(stroke):
    '''
    Make the start of the stroke at the center of the bit map image
    '''
    # how to get to the start of the brush stroke from the top left of the cut out region
    down = 0.5
    right = 0.2

    stroke, down, right = crop_stroke(stroke.copy(), down, right)

    # Padding for rotation. Ensure start of brush stroke is centered in square image
    h, w = stroke.shape

    # PadX to center the start of stroke
    padX = max(0, w*(1-2*right)), max(0, w*(2*right-1))
    # PadY to center the start of the stroke
    padY = max(0, h*(1-2*down)), max(0, h*(2*down - 1))

    # Pad to become square
    newW, newH = padX[0] + padX[1] + w, padY[0] + padY[1] + h

    if newH > newW:
        xtra_x = (newH - newW)/2
        padX = padX[0] + xtra_x, padX[1] + xtra_x
    elif newH < newW:
        xtra_y = (newW - newH)/2
        padY = padY[0] + xtra_y, padY[1] + xtra_y
    padX = int(padX[0]), int(padX[1])
    padY = int(padY[0]), int(padY[1])

    imgP = np.pad(stroke, [padY, padX], 'constant')
    return imgP


# def export_strokes(opt):
#     strokes = pickle.load(gzip.open((os.path.join(opt.cache_dir, "strokes.pkl")),'rb'))

#     strokes = [center_stroke(s) for s in strokes]
#     strokes_processed = []

#     w, h = opt.CANVAS_WIDTH_PIX, opt.CANVAS_HEIGHT_PIX
#     # print(opt.CANVAS_WIDTH_PIX, opt.CANVAS_HEIGHT_PIX)
#     for s in strokes:
#         #t = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.uint8)
#         #t[:,:,3] = (s*255.).astype(np.uint8)
#         t = (s*255.).astype(np.uint8)
#         # plt.imshow(t)
#         # plt.scatter(t.shape[1]/2, t.shape[0]/2)
#         # plt.show()

#         # Make the stroke centered on a full size canvas
#         padX = int((w - t.shape[1])/2)
#         padY = int((h - t.shape[0])/2)

#         # In case of odd numbers
#         xtra_x, xtra_y = w - (padX*2+t.shape[1]), h - (padY*2+t.shape[0])
#         #print(t.shape, padX, padY, xtra_x, xtra_y)
#         full_size = np.pad(t, [(padY,padY+xtra_y), (padX,padX+xtra_x)], 'constant')
#         t = np.zeros((full_size.shape[0], full_size.shape[1], 4), dtype=np.uint8)
#         t[:,:,3] = full_size.astype(np.uint8)
#         t[:,:,2] = 200 # Some color for fun
#         t[:,:,1] = 120

#         strokes_processed.append(t)


#     # h, w = int(opt.CANVAS_HEIGHT_PIX/scale_factor), int(opt.CANVAS_WIDTH_PIX/scale_factor)
#     # canvas = torch.zeros((h, w, 3))

#     # strokes_processed = []
#     # for s in strokes:
#     #     s = cv2.resize(s[:,:,3], (int(s.shape[1]/scale_factor), int(s.shape[0]/scale_factor)))

#     #     padX = int((w - s.shape[1])/2)
#     #     padY = int((h - s.shape[0])/2)

#     #     # In case of odd numbers
#     #     xtra_x, xtra_y = w - (padX*2+s.shape[1]), h - (padY*2+s.shape[0])
#     #     s = np.pad(s, [(padY,padY+xtra_y), (padX,padX+xtra_x)], 'constant')
#     #     t = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.uint8)
#     #     t[:,:,3] = s.astype(np.uint8)
#     #     t[:,:,2] = 200 # Some color for fun
#     #     t[:,:,1] = 120

#     #     strokes_processed.append(torch.from_numpy(t).float().to(device) / 255.)
#     # return strokes_processed




#     #np.save(open('strokes_numpy.npy', 'wb'), strokes_processed)
#     pickle.dump(strokes_processed, gzip.open(os.path.join(opt.cache_dir, 'strokes_centered.npy'), 'wb'))