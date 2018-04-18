import cv2 as cv
import numpy as np
import histogram
import segmentation as sg


class HistogramSegmentation(sg.Segmentation):

    def __init__(self, segment_width, segment_height, bin_threshold=225, histogram_threshold=0.1):
        sg.Segmentation.__init__(self, segment_width, segment_height, bin_threshold)
        self.histogram = histogram.Histogram(histogram_threshold)

    def _get_contours(self, input_img, lines, words_per_line):
        word_rects = []

        for line, words in zip(lines, words_per_line):
            for word in words:
                mod_img, contours, hierarchy = cv.findContours(input_img[line[0]:line[1], word[0]:word[1]],
                                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                letters = []
                for contour in contours:
                    letters.append(cv.boundingRect(contour))
                letters = cv.groupRectangles(letters, 0)
                word_rects.append((line[0], line[1], word[0], word[1], letters[0]))

        return word_rects

    def _resize_letter(self, rects, sub_images):
        resized_words = []

        for word in rects:
            resized_letters = []
            y0, y1, x0, x1, letters = word
            img = sub_images[y0:y1, x0:x1]
            for letter in letters:
                x, y, width, height = tuple(letter)
                cropped = img[y:y + height, x:x + width]
                cropped = cv.resize(cropped, (self.resized_height, self.resized_width),
                                    interpolation=cv.INTER_CUBIC)
                resized_letters.append(cropped)
            resized_words.append((y0, y1, x0, x1, resized_letters))

        return resized_words

    def _thin(self, images):
        for word in range(len(images)):
            for letter in range(len(images[word][-1])):
                images[word][-1][letter] = cv.ximgproc.thinning(images[word][-1][letter])

        return  images

    def segment(self, input_img, is_gray=False):
        binarized = self._binarize(input_img, is_gray)[1]
        lines = self.histogram.find_lines(binarized)
        words_per_line = self.histogram.find_words(binarized, lines)
        word_rects = self._get_contours(binarized, lines, words_per_line)
        resized = self._resize_letter(word_rects, binarized)

        return self._thin(resized), binarized