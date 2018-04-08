import cv2 as cv
from util import is_white


class FeatureExtraction(object):

    def __init__(self, step):
        self.step = step

    def _count_white_pixels(self, img):
        vector = []

        for i in range(0, img.shape[0], self.step):
            for j in range(0, img.shape[1], self.step):
                count = 0
                for y in range(i, i + self.step):
                    for x in range(j, j + self.step):
                        if is_white(img[y, x]):
                            count += 1
                vector.append(count)
        return vector

    def extract(self, images):
        return [self._count_white_pixels(img) for img in images]