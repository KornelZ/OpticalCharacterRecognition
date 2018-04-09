import cv2 as cv
import segmentation as sg
import feature_extraction as fe
import classifier as cl
from util import get_labels

DIM = 40
EXTRACTION_STEP = 4


def preprocess(input, segmentation, extraction, dim):
    segmented = segmentation.segmentize(input, False)
    feature_vector = extraction.extract(segmented)
    labels = get_labels(input.shape[1], input.shape[0], dim, dim)
    return [feature_vector, labels]


src = cv.imread("letters.png")
train = src[:, :src.shape[1] // 2]
test = src[:, src.shape[1] // 2:]

seg = sg.Segmentation(DIM, DIM)
ext = fe.FeatureExtraction(EXTRACTION_STEP)


training_input = preprocess(train, seg, ext, DIM)
test_input = preprocess(test, seg, ext, DIM)

svm = cl.Classifier(cl.CLASS_BAYES)
svm.train(training_input)
result = svm.predict(test_input)

print(result)

count = 0
for label, predicted in zip(test_input[1], result):
    if label == int(predicted[0]):
        count += 1
print(count / len(test_input[1]))


