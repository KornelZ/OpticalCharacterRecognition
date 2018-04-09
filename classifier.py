import cv2 as cv
import numpy as np


class Classifier(object):

    def __init__(self, classifier):
        if classifier is "SVM":
            self.classifier = cv.ml.SVM_create()
            self.classifier.setType(cv.ml.SVM_C_SVC)
            self.classifier.setKernel(cv.ml.SVM_INTER) #75% accuracy, no params required
            #45$ accuracy for linear kernel


    def train(self, input):
        train_arr = np.array(input[0], np.float32)
        label_arr = np.array(input[1])

        self.classifier.train(train_arr, cv.ml.ROW_SAMPLE,
                              label_arr)

    def predict(self, input):
        predict_arr = np.array(input[0], np.float32)

        return self.classifier.predict(predict_arr)[1]