import cv2 as cv
import numpy as np
from util import log

CLASS_SVM = 'svm'
CLASS_BAYES = "bayes"
CLASS_RTREES = "rtrees"
CLASS_ANN = "ann"

class Classifier(object):

    RTREE_DEPTH = 12
    MIN_SAMPLE_COUNT = 40
    EPOCHS = 1400
    SIGMA_ALPHA = 2
    SIGMA_BETA = 1

    def __init__(self, classifier, in_hidden_layer_size=0, out_layer_size=0):
        self.out_layer_size = out_layer_size
        self.classifier_name = classifier

        if classifier == CLASS_SVM:
            self.classifier = cv.ml.SVM_create()
            self.classifier.setType(cv.ml.SVM_C_SVC)
            self.classifier.setKernel(cv.ml.SVM_INTER) #94% accuracy,
            #45$ accuracy for linear kernel
        elif classifier == CLASS_BAYES:
            self.classifier = cv.ml.NormalBayesClassifier_create() #85%
        elif classifier == CLASS_RTREES: #92
            self.classifier = cv.ml.RTrees_create()
            self.classifier.setMaxDepth(Classifier.RTREE_DEPTH)
            self.classifier.setMinSampleCount(Classifier.MIN_SAMPLE_COUNT)
        elif classifier == CLASS_ANN: #94
            layers = np.array([in_hidden_layer_size,
                               in_hidden_layer_size,
                               in_hidden_layer_size,
                               out_layer_size])
            self.classifier = cv.ml.ANN_MLP_create()
            self.classifier.setLayerSizes(layers)
            self.classifier.setTermCriteria((cv.TermCriteria_COUNT, Classifier.EPOCHS, 0))
            self.classifier.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM,
                                                  Classifier.SIGMA_ALPHA, Classifier.SIGMA_BETA)

    @log
    def train(self, input):
        if self.classifier_name == CLASS_ANN:
            label_arr = np.array(self._one_hot_encode(input[1]), np.float32)
        else:
            label_arr = np.array(input[1])
        train_arr = np.array(input[0], np.float32)

        self.classifier.train(train_arr, cv.ml.ROW_SAMPLE, label_arr)

    @log
    def predict(self, input):
        predict_arr = np.array(input[0], np.float32)

        if self.classifier_name == CLASS_BAYES:
            return self.classifier.predictProb(predict_arr)[1]

        prediction = self.classifier.predict(predict_arr)[1]
        if self.classifier_name == CLASS_ANN:
            return self._one_hot_decode(prediction)
        return prediction

    @log
    def load(self, use_letters):
        self.classifier = self.classifier.load(self._get_name(use_letters))

    @log
    def save(self, use_letters):
        self.classifier.save(self._get_name(use_letters))

    def _get_name(self, use_letters):
        if use_letters:
            data_name = "letters"
        else:
            data_name = "digits"

        return "models\\" + data_name + "_" + self.classifier_name + ".xml"

    @log
    def _one_hot_encode(self, labels):
        for i in range(len(labels)):
            tmp = [0] * self.out_layer_size
            tmp[labels[i]] = 1
            labels[i] = tmp

        return labels

    @log
    def _one_hot_decode(self, labels):
        decoded = np.zeros((labels.shape[0], 1), dtype=np.int32)
        for i in range(len(labels)):
            decoded[i] = np.argmax(labels[i])

        return decoded
