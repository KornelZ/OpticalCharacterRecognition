import cv2 as cv
import segmentation as sg
import feature_extraction as fe

src = cv.imread("letters.png")
seg = sg.Segmentation(40, 40)
segmented = seg.segmentize(src, False)
extraction = fe.FeatureExtraction(8)
feature_vector = extraction.extract(segmented)
print(feature_vector[0])