import cv2
cv2.__version__
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv2.imread('resources/dog.jpg')
img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

# set the lower and upper threshold
med_val = np.median(img_orig)
lower = int(max(0, .7 * med_val))
upper = int(min(255, 1.3 * med_val))

# blurring with ksize = 3
img_k3 = cv2.blur(img_orig, ksize=(3, 3))
# canny detection with different thresholds
edges_k3 = cv2.Canny(img_k3, threshold1=lower, threshold2=upper)
edges_k3_2 = cv2.Canny(img_k3, lower, upper + 75)

# blurring with ksize = 5
img_k5 = cv2.blur(img_orig, ksize=(5, 5))
# canny detection with different thresholds
edges_k5 = cv2.Canny(img_k5, lower, upper)
edges_k5_2 = cv2.Canny(img_k5, lower, upper + 75)

# plot the images
images = [edges_k3, edges_k3_2, edges_k5, edges_k5_2]
titles = ['blurring, kernel = 3x3, lower threshold',
          'blurring, kernel = 3x3, higher threshold',
          'blurring, kernel = 5x5, lower threshold',
          'blurring, kernel = 5x5, higher threshold']
plt.figure(figsize=(20, 15))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
    plt.axis('off')
    plt.title(titles[i])

plt.show()