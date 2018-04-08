import cv2 as cv
import numpy as np

class Classifier(object):

    def __init__(self, classifier):

        if classifier is "SVM":
            self.classifier = cv.ml.SVM_create()


    def train(self, input):
        train_arr = np.array(input[0], np.float32)
        label_arr = np.array(input[1])

        self.classifier.train(train_arr, cv.ml.ROW_SAMPLE, label_arr)
        self.classifier.save("svm_train.dat")