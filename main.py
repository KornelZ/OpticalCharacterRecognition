import cv2 as cv
import segmentation as sg
import histogram_segmentation as hist
import feature_extraction as fe
import classifier as cl
import numpy as np
from util import get_labels

DIM = 40
EXTRACTION_STEP = 4


def preprocess(input, segmentation, extraction, dim):
    segmented = segmentation.segment(input)
    feature_vector = extraction.extract(segmented)
    labels = get_labels(input.shape[1], input.shape[0], dim, dim)
    return [feature_vector, labels]


src = cv.imread("data\letters.png")
train = src[:, :4 * src.shape[1] // 2]
test = src[:, 4 * src.shape[1] // 5:]

seg = sg.Segmentation(DIM, DIM, 225)
ext = fe.FeatureExtraction(EXTRACTION_STEP)


training_input = preprocess(train, seg, ext, DIM)
test_input = preprocess(test, seg, ext, DIM)
classifier = cl.Classifier(cl.CLASS_BAYES)
classifier.train(training_input)
result = classifier.predict(test_input)

count = 0
for label, predicted in zip(test_input[1], result):
    if label == int(predicted[0]):
        count += 1
print(count / len(test_input[1]))

"""
src = cv.imread("data/in2.png")
histogram = hist.HistogramSegmentation(DIM, DIM, bin_threshold=240, histogram_threshold=0)
words, b = histogram.segment(src)

collage = None
run_first = True
count = 0
imgs = []
for w in words:
    y0, y1, x0, x1, letters = w
    for l in letters:
        count += 1
        imgs.append(l)

feature_vectors = ext.extract(imgs)

result = classifier.predict((feature_vectors, 0))

print(result)"""




