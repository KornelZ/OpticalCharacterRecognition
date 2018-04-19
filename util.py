import functools


def log(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print("At " + function.__name__)
        return function(*args, **kwargs)

    return wrapper


def is_white(pixel):
    return pixel == 255

@log
def get_labels(img_width, img_height, segment_width, segment_height, use_letters):
    cols = img_width // segment_width
    rows = img_height // segment_height
    labels = []

    for i in range(rows):
        for j in range(cols):
            if use_letters:
                labels.append(i)
            else:
                labels.append(i // 5)

    return labels
