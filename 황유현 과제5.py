import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy
from numpy.core.fromnumeric import reshape

# imread returned bgr
img = cv.imread('dgu_class\image_processing\dgu_night_color.png')

# Y = img_R*0.257 + img_G*0.504 + img_B*0.098
# Y = np.array(Y, dtype=np.uint8)
# Cb = img_R*-0.148 + img_G*-0.294 + img_B*-0.493
# Cb = np.array(Y, dtype=np.uint8)
# Cr = img_R*0.439 + img_G*-0.368 + img_B*-0.071
# Cr = np.array(Y, dtype=np.uint8)

# original image
img_org = copy.deepcopy(img)

# YCrCb
img_in = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

cv.imshow('img_in', img_in)

# Y_in
Y_in = img_in[:,:,0]

cv.imshow('Y_in', Y_in)

def get_hist_eql(image):
    height, width = image.shape

    new_image = np.zeros_like(image)
    
    hist = np.zeros(256)
    cdf = np.zeros(256)

    # each label
    for x in range(width):
        for y in range(height):
            hist[image[y,x]] += 1
            # print(img[y,x])

    # hist = hist/(height*width)

    for i in range(256):
        cdf[i] += (cdf[i-1]+hist[i])

    cdf = cdf/(height*width)

    # print(cdf)
    
    output_hist = np.zeros(256)

    for i in range(256):
        output_hist[i] = np.round((cdf[i] * 255),0)

    # print(output_hist)

    for x in range(width):
        for y in range(height):
            new_image[y,x] = output_hist[image[y,x]]

    return new_image

# Y_out
Y_out = get_hist_eql(Y_in)

cv.imshow('Y_out', Y_out)

# Y_in = 1/Y_in

s = 0.4

img_in = np.array(img_in, dtype=np.float32)

img_in[:,:,0] = (img[:,:,0]/Y_in)**s
img_in[:,:,1] = (img[:,:,1]/Y_in)**s
img_in[:,:,2] = (img[:,:,2]/Y_in)**s

# dtype uint8
img_out = np.zeros_like(img_in)

img_out[:,:,0] = img_in[:,:,0]*Y_out
img_out[:,:,1] = img_in[:,:,1]*Y_out
img_out[:,:,2] = img_in[:,:,2]*Y_out

# print(img_out[:,:,0])
# print(img_out[:,:,1])
# print(img_out[:,:,2])

h, w = 720, 1280
for x in range(w):
    for y in range(h):
        if img_out[y,x,0]>255:
            img_out[y,x]=255
        if img_out[y,x,1]>255:
            img_out[y,x]=255
        if img_out[y,x,2]>255:
            img_out[y,x]=255

# img_out index to uint8
img_out = np.array(img_out, dtype=np.uint8)
        
cv.imshow('img_out', img_out)
cv.waitKey()