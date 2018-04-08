import cv2 as cv
import segmentation as sg
import feature_extraction as fe
import classifier
from util import get_labels

src = cv.imread("letters.png")
seg = sg.Segmentation(40, 40)
segmented = seg.segmentize(src, False)
extraction = fe.FeatureExtraction(8)
feature_vector = extraction.extract(segmented)
labels = get_labels(src.shape[1], src.shape[0], 40, 40)

print("TRAINING")
training_input = [feature_vector, labels]
svm = classifier.Classifier("SVM")
svm.train(training_input)