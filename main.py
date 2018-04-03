import cv2 as cv
import segmentation as sg

src = cv.imread("letters.png")
seg = sg.Segmentation(40, 40)
resized = seg.segmentize(src, False)
cv.imshow("t", resized[0])
cv.waitKey()