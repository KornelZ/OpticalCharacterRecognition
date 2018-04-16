import cv2 as cv
import numpy as np

CLASS_SVM = 0
CLASS_BAYES = 1
CLASS_RTREES = 2
CLASS_ANN = 3


class Classifier(object):

    RTREE_DEPTH = 16
    MIN_SAMPLE_COUNT = 40

    def __init__(self, classifier):
        self.classifier_name = classifier
        if classifier is CLASS_SVM:
            self.classifier = cv.ml.SVM_create()
            self.classifier.setType(cv.ml.SVM_C_SVC)
            self.classifier.setKernel(cv.ml.SVM_INTER) #80% accuracy, no params required
            #45$ accuracy for linear kernel
        elif classifier is CLASS_BAYES:
            self.classifier = cv.ml.NormalBayesClassifier_create() #80%
        elif classifier is CLASS_RTREES:
            self.classifier = cv.ml.RTrees_create()
            self.classifier.setMaxDepth(Classifier.RTREE_DEPTH)
            self.classifier.setMinSampleCount(Classifier.MIN_SAMPLE_COUNT)

    def train(self, input):
        train_arr = np.array(input[0], np.float32)
        label_arr = np.array(input[1])

        self.classifier.train(train_arr, cv.ml.ROW_SAMPLE,
                              label_arr)

    def predict(self, input):
        predict_arr = np.array(input[0], np.float32)

        if self.classifier_name is CLASS_BAYES:
            return self.classifier.predictProb(predict_arr)[1]

        return self.classifier.predict(predict_arr)[1]
