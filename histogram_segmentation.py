import cv2 as cv
import numpy as np
import segmentation as sg


class HistogramSegmentation(sg.Segmentation):

    THRESHOLD = 0.1

    def _get_histogram(self, input_img, dimension):
        """dimension == 0 -> get histogram of rows
                        1 -> get histogram of columns"""
        t = input_img.max()
        input_dim = 0
        if dimension == 0:
            input_dim = 1
        histogram = cv.reduce(input_img, input_dim, cv.REDUCE_SUM, dtype=cv.CV_32F)
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
            if histogram[i] > HistogramSegmentation.THRESHOLD:
                if start == 0:
                    start = i
                end = i
            elif start != 0:
                lines.append((start, end))
                start = 0
                end = 0
        return lines

    def _find_max_space(self, histogram):
        max_len = 0
        curr_len = 0
        found_first = False
        for i in range(len(histogram)):
            if histogram[i] < HistogramSegmentation.THRESHOLD\
                    and found_first:
                curr_len += 1
            elif histogram[i] > HistogramSegmentation.THRESHOLD:
                if curr_len > max_len:
                    max_len = curr_len
                curr_len = 0
                found_first = True

        return max_len // 2

    def _divide_into_words(self, histogram, max_space):
        words = []
        start = 0
        word_len = 0
        found_word = False


    def _find_words(self, input_img, lines):
        words_per_line = []
        for y1, y2 in lines:
            histogram = self._get_histogram(input_img[y1:y2, :], dimension=1)
            max_space = self._find_max_space(histogram)

        return words_per_line

    def segment(self, input_img, is_gray=False):
        binarized = self._binarize(input_img, is_gray)[1]
        lines = self._find_lines(binarized)
        words_per_line = self._find_words(binarized, lines)

        return lines