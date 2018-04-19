import cv2 as cv
import histogram_segmentation as hist
import feature_extraction as fe
import classifier as cl

LETTER_SIZE = 40
DIGIT_SIZE = 20
EXTRACTION_STEP = 4
BIN_THRESHOLD = 140


def show_result(src, rects, result):
    i = 0
    for word in rects:
        t = ""
        for _ in word[-1]:
            t += chr(int(result[i]) + 65)
            i += 1
        cv.rectangle(src, (word[2], word[0]), (word[3], word[1]), (255, 0, 0), 3)
        cv.putText(src, t, (word[2], word[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    cv.imshow("t", src)
    cv.waitKey()


def predict(path, use_letters, used_classifier):

    src = cv.imread("data\\" + path)
    if use_letters:
        dim = LETTER_SIZE
    else:
        dim = DIGIT_SIZE
    histogram = hist.HistogramSegmentation(dim, dim, BIN_THRESHOLD, histogram_threshold=0)
    ext = fe.FeatureExtraction(EXTRACTION_STEP)
    words, b, rects = histogram.segment(src)

    imgs = []
    for w in words:
        y0, y1, x0, x1, letters = w
        for l in letters:
            imgs.append(l)

    feature_vectors = ext.extract(imgs)
    classifier = cl.Classifier(used_classifier, 64, 32)
    classifier.load(use_letters)
    result = classifier.predict((feature_vectors, 0))
    show_result(src, rects, result)


def main():
    path = input("Path to file:")
    use_letters = input("Letters or digits?")
    used_classifier = input("Choose classifier: svm, bayes, rtrees, ann")
    if use_letters == "letters":
        letters = True
    else:
        letters = False
    predict(path, letters, used_classifier)


if __name__ == "__main__":
    main()
