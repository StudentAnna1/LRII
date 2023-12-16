import cv2
import numpy as np


# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("test_foto.jpg")
# DISPLAY
print(img.shape)
imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)
imgCropped = img[463:1191,352:495]
cv2.imshow("Image",img)
#cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)
