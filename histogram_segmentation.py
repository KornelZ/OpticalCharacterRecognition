import cv2 as cv
import numpy as np
import segmentation as sg


class HistogramSegmentation(sg.Segmentation):

    def _get_histogram(self, input_img, dimension):
        """dimension == 0 -> get histogram of rows
                        1 -> get histogram of columns"""
        input_dim = 0
        if dimension == 0:
            input_dim = 1
        histogram = np.zeros(input_img.shape[dimension], np.float32)
        cv.reduce(input_img, input_dim, cv.REDUCE_SUM, histogram,
                  cv.CV_32F)
        total = input_img.shape[input_dim] * 255;
        for i in range(len(histogram)):
            histogram[i] /= total

        return histogram

    def _find_lines(self, input_img):
        histogram = self._get_histogram(input_img, dimension=0)
        lines = []
        start = 0
        end = 0
        for i in range(len(histogram)):
            if histogram[i] > 0.1:
                if start == 0:
                    start = i
                end = i
            elif start != 0:
                lines.append((start, end))
                start = 0
                end = 0
        return lines

    def segment(self, input_img, is_gray=False):
        binarized = self._binarize(input_img, is_gray)[1]
        lines = self._find_lines(binarized)
        return lines