import cv2 as cv
import numpy as np

img = cv.imread("gifs/Untitled2.jpeg",cv.IMREAD_COLOR)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, mask = cv.threshold(img_gray, 1, 255, cv.THRESH_BINARY_INV)
cv.imwrite("gifs/mask.jpg",mask)

restored = cv.inpaint(img, mask, 8 ,cv.INPAINT_TELEA)
cv.imwrite("gifs/restored.jpg",restored)

