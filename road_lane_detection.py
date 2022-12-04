import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg


def grey(image):
    # convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image, 40, 200)
    return edges


def region_of_interest(image):
    # get the image dimensions
    rows, cols = image.shape

    # parameters for the image mask and for filtering lines
    left = 0.02 * cols
    right = 0.98 * cols
    x_margin = 0.0 * cols
    bottom = 0.85 * rows
    top = 0.40 * rows
    top_width = 0.20 * cols
    top_left_margin = 0.50 * cols
    top_right_margin = 0.0 * cols

    # a mask to be applied to the image
    top_left = (left + right - top_width - top_left_margin) / 2
    top_right = (left + right + top_width + top_right_margin) / 2
    vertices = np.array([[
        (left - x_margin, bottom),
        (top_left, top),
        (top_right, top),
        (right + x_margin, bottom)
    ]],
        dtype=np.int32)

    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, vertices, 255)
    mask = cv2.bitwise_and(image, mask)

    return mask


# def color_mask(image):
#     img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     lower_gray = np.array([220, 25, 28])
#     upper_gray = np.array([64, 2, 91])
#
#     # mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
#     # mask_white = cv2.inRange(grey(image), 200, 255)
#     # mask_gray_white = cv2.bitwise_or(mask_white, mask_gray)
#     # img_gray_white = cv2.bitwise_and(grey(image), mask_gray_white)
#
#     mask_gray = cv2.inRange(img_hsv, lower_gray, upper_gray)
#     img_gray_masked = cv2.bitwise_and(image, image, mask_gray)
#     return img_gray_masked


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                print(f"x1: {x1}")
                print(f"y1: {y1}")
                print(f"x2: {x2}")
                print(f"y2: {y2}")
                # draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def average(image, lines):
    left = []
    right = []
    left_line = None
    right_line = None
    left_slope = None
    right_slope = None

    if lines is not None:
        for line in lines:
            print(f"line: {line}")
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            print(f"parameters: {parameters}")
            slope = parameters[0]
            y_int = parameters[1]
            # lines on the right have positive slope, and lines on the left have neg slope
            if 1e-2 < abs(slope) < 1e2:  # Protection against giant numbers
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))

    if left:
        # takes average among all the columns (column0: slope, column1: y_int)
        left_avg = np.average(left, axis=0)
        # create lines based on averages calculates
        print(f"left_avg: {left_avg}")
        left_line = make_points(image, left_avg)
        left_slope = math.degrees(left_avg[0])

    if right:
        # takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        # create lines based on averages calculates
        print(f"right_avg: {right_avg}")
        right_line = make_points(image, right_avg)
        right_slope = math.degrees(right_avg[0])

    return np.array([left_line, right_line, left_slope, right_slope], dtype=object)


def make_points(image, average):
    # print(f"average: {average}")
    slope, y_int = average
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3 / 5))
    print(f"y1: {y1}")
    print(f"y2: {y2}")
    print(f"y1 - y_int: {y1 - y_int}")
    print(f"y2 - y_int: {y2 - y_int}")
    # determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def hough_transform(image):
    rho = 3  # The resolution of the parameter r in pixels
    threshold = 100  # The minimum number of intersections to "*detect*" a line
    min_line_length = 40  # minimum number of points that can form a line. Lines with less than this number of points are disregarded
    max_line_gap = 5  # maximum gap between two points to be considered in the same line.

    # DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
    lines = cv2.HoughLinesP(image,
                            rho,
                            np.pi / 180,
                            threshold,
                            np.array([]),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines


def get_degrees(averaged_lines, image_shape):
    degrees = 0
    left_line, right_line, degrees_left_slope, degrees_right_slope = averaged_lines
    if left_line is not None and right_line is not None:
        left_point = (left_line[0], left_line[1])
        right_point = (right_line[0], right_line[1])
        mid_x, mid_y = midpoint(left_point, right_point)
        print(f"mid_x: {mid_x}, mid_y: {mid_y}")

        degrees = 90 + math.degrees(math.atan2(mid_y - image_shape[1], mid_x - image_shape[0] / 2))

    if left_line is not None:
        degrees = abs(degrees_left_slope)

    if right_line is not None:
        degrees = -abs(degrees_right_slope)
    return degrees


def process_image(image, draw_image=False):
    blur = gauss(image)
    gray = grey(blur)
    edges = canny(gray)
    isolated = region_of_interest(edges)

    lines = hough_transform(isolated)
    averaged_lines = average(image, lines)
    print(f"averaged_lines: {averaged_lines}")

    degrees = get_degrees(averaged_lines, image.shape)
    print(f"degrees: {degrees}")

    if draw_image:
        # Visual part
        black_lines = display_lines(image, averaged_lines[0:1])

        # taking wighted sum of original image and lane lines image
        lanes = cv2.addWeighted(image, 0.8, black_lines, 1, 1)
        return lanes

    return degrees


def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2,
            (point1[1] + point2[1]) / 2)


if __name__ == "__main__":
    # Test images
    import os

    test_image_dir = "./resources/"
    test_images = os.listdir(test_image_dir)

    for img in test_images:
        print(img)
        image = mpimg.imread(test_image_dir + img)

        # Steps from process_image
        blur = gauss(image)
        gray = grey(blur)
        edges = canny(gray)
        isolated = region_of_interest(edges)

        images = [
            [image, f"{img}: Base img"],
            [region_of_interest(grey(image)), f"{img}: ROI"],
            [region_of_interest(edges), f"{img}: Edge detect"],
            [process_image(image, True), f"{img}: Lines"],
        ]

        plt.figure(figsize=(20, 15))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i][0])
            plt.axis('off')
            plt.title(images[i][1])

        plt.show()
        print("\n")
