import cv2 as cv

src = cv.imread("input.jpg")
grayscale = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
_, binarized = cv.threshold(grayscale, 128, 255, cv.THRESH_BINARY_INV)
thinned = cv.ximgproc.thinning(binarized)
cv.imshow("Grayscale", grayscale)
cv.imshow("Binarized", binarized)
cv.imshow("Skeletonized", thinned)
cv.waitKey()