import cv2
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv2.imread('resources/center_2022_11_14_10_22_32_396.jpg')
img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Original image', img_orig)
# cv2.imshow('Gray image', img_gray)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def edge_detect_images(img, kernel_size: tuple, color_space: str):
    # set the lower and upper threshold
    med_val = np.median(img)
    lower = int(max(0, .8 * med_val))
    upper = int(min(255, 1.3 * med_val))
    margin = 75
    upper_higher = upper + margin

    # blurring with ksize = 5
    img_k3 = cv2.blur(img, ksize=kernel_size)
    # canny detection with different thresholds
    edges_k3 = cv2.Canny(img_k3, threshold1=lower, threshold2=upper)
    edges_k3_2 = cv2.Canny(img_k3, lower, upper_higher)

    # Gaussian blurring with ksize = 5
    img_k5 = cv2.GaussianBlur(img, ksize=kernel_size, sigmaX=0)
    # canny detection with different thresholds
    edges_k5 = cv2.Canny(img_k5, lower, upper)
    edges_k5_2 = cv2.Canny(img_k5, lower, upper_higher)

    # Return the edge detected images
    return [
        [edges_k3, edges_k3_2, edges_k5, edges_k5_2],
        [f'[{color_space}] blurring, kernel = {kernel_size[0]}x{kernel_size[1]}, threshold = {lower} | {upper}',
         f'[{color_space}] blurring, kernel = {kernel_size[0]}x{kernel_size[1]}, threshold = {lower} | {upper_higher}',
         f'[{color_space}] Gaussian blurring, kernel = {kernel_size[0]}x{kernel_size[1]}, threshold = {lower} | {upper}',
         f'[{color_space}] Gaussian blurring, kernel = {kernel_size[0]}x{kernel_size[1]} threshold = {lower} | {upper_higher}',]
    ]


# plot the images - Colored
output_color = edge_detect_images(img_orig, (5, 5), 'RGB')
images_color = output_color[0]
titles = output_color[1]
plt.figure(figsize=(20, 15))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images_color[i])
    plt.axis('off')
    plt.title(titles[i])

plt.show()


# plot the images - Gray
output_gray = edge_detect_images(img_gray, (5, 5), 'Gray')
images_gray = output_gray[0]
titles = output_gray[1]
plt.figure(figsize=(20, 15))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images_gray[i])
    plt.axis('off')
    plt.title(titles[i])

plt.show()
