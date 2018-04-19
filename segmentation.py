import cv2 as cv
from util import log


class Segmentation(object):

    def __init__(self, segment_width, segment_height, bin_threshold=225):
        self.segment_width = segment_width
        self.segment_height = segment_height
        self.resized_width = 32
        self.resized_height = 32
        self.bin_threshold = bin_threshold

    @log
    def _binarize(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return cv.threshold(img, self.bin_threshold, 255, cv.THRESH_BINARY_INV)

    @log
    def _subdivide(self, img):
        sub_images = []
        for y in range(0, img.shape[0], self.segment_height):
            for x in range(0, img.shape[1], self.segment_width):
                sub_images.append(img[y:y + self.segment_height, x:x + self.segment_width])
        return sub_images

    @log
    def _bound_letter(self, sub_images):
        bound_rects = []
        for img in sub_images:
            img_rect = []
            mod_img, contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for contour in contours:
                img_rect.append(cv.boundingRect(contour))
            img_rect = cv.groupRectangles(img_rect, 0)
            bound_rects.append(img_rect[0])
        return bound_rects

    @log
    def _resize_letter(self, rects, sub_images):
        resized = []
        for img, rect in zip(sub_images, rects):
            dims = rect.tolist()[0]
            x, y, width, height = tuple(dims)
            cropped = img[y:y + height, x:x + width]
            cropped = cv.resize(cropped, (self.resized_height, self.resized_width), interpolation=cv.INTER_CUBIC)
            resized.append(cropped)
        return resized

    @log
    def _thin(self, images):
        return [cv.ximgproc.thinning(img) for img in images]

    @log
    def segment(self, input_img):
        binarized = self._binarize(input_img)[1]
        sub_images = self._subdivide(binarized)
        bound_rects = self._bound_letter(sub_images)
        resized = self._resize_letter(bound_rects, sub_images)
        return self._thin(resized)


