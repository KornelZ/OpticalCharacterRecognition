import cv2 as cv

class Histogram(object):

    def __init__(self, threshold):
        self.threshold = threshold

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

    def find_lines(self, input_img):
        histogram = self._get_histogram(input_img, dimension=0)
        lines = []
        start = 0
        end = 0
        for i in range(len(histogram)):
            if histogram[i] > self.threshold:
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
        for i in range(len(histogram[0])):
            if histogram[0, i] < self.threshold and found_first:
                curr_len += 1
            elif histogram[0, i] >= self.threshold:
                if curr_len > max_len:
                    max_len = curr_len
                curr_len = 0
                found_first = True

        return max_len // 2

    def _divide_into_words(self, histogram, max_space):
        words = []
        start = 0
        word_len = 0
        space_len = 0
        found_word = False
        for i in range(len(histogram[0])):
            if space_len >= max_space and found_word:
                space_len = 0
                found_word = False
                words.append((start, start + word_len))
                start = 0
                word_len = 0
            if histogram[0, i] < self.threshold:
                space_len += 1
            elif histogram[0, i] >= self.threshold:
                if not found_word:
                    start = i
                found_word = True
                word_len += 1
                space_len = 0

        return words

    def find_words(self, input_img, lines):
        words_per_line = []
        for y1, y2 in lines:
            histogram = self._get_histogram(input_img[y1:y2, :], dimension=1)
            max_space = self._find_max_space(histogram)
            words_per_line.append(self._divide_into_words(histogram, max_space))

        return words_per_line
