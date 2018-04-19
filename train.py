import cv2 as cv
import segmentation as sg
import feature_extraction as fe
import classifier as cl
from util import get_labels

LETTER_SIZE = 40
DIGIT_SIZE = 20
EXTRACTION_STEP = 4
IN_HIDDEN_LAYER_SIZE = 64
BIN_THRESHOLD = 225


def preprocess(input, segmentation, extraction, use_letters, dim):
    segmented = segmentation.segment(input)
    feature_vector = extraction.extract(segmented)
    labels = get_labels(input.shape[1], input.shape[0], dim, dim, use_letters)

    return [feature_vector, labels]


def train(path, use_letters, used_classifier, dim):
    src = cv.imread("data\\" + path)
    train = src[:, :4 * src.shape[1] // 2]
    test = src[:, 4 * src.shape[1] // 5:]

    seg = sg.Segmentation(dim, dim, BIN_THRESHOLD)
    ext = fe.FeatureExtraction(EXTRACTION_STEP)

    training_input = preprocess(train, seg, ext, use_letters, dim)
    test_input = preprocess(test, seg, ext, use_letters, dim)

    if use_letters:
        out_layer_size = 26
    else:
        out_layer_size = 10

    classifier = cl.Classifier(used_classifier,
                               in_hidden_layer_size=IN_HIDDEN_LAYER_SIZE, out_layer_size=out_layer_size)
    classifier.train(training_input)
    result = classifier.predict(test_input)

    count = 0
    for label, predicted in zip(test_input[1], result):
        if label == int(predicted[0]):
            count += 1
    print(count / len(test_input[1]))
    classifier.save(use_letters)


def main():
    path = input("Path to file:")
    use_letters = input("Letters or digits?")
    used_classifier = input("Choose classifier: svm, bayes, rtrees, ann")
    if use_letters == "letters":
        letters = True
        dim = LETTER_SIZE
    else:
        letters = False
        dim = DIGIT_SIZE
    train(path, letters, used_classifier, dim)
    print("Finished")


if __name__ == "__main__":
    main()


