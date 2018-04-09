


def is_white(pixel):
    return pixel == 255


def get_labels(img_width, img_height, segment_width, segment_height):
    cols = img_width // segment_width
    rows = img_height // segment_height
    labels = []

    for i in range(rows):
        for j in range(cols):
            labels.append(i)
    return labels
