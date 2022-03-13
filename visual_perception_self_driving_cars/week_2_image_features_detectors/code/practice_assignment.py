import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *

np.random.seed(1)
# np.set_printoptions(threshold=np.nan)

dataset_handler = DatasetHandler()

image = dataset_handler.images[0]
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')

image_rgb = dataset_handler.images_rgb[0]
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image_rgb)

depth = dataset_handler.depth_maps[0]
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(depth, cmap='jet')

print("Depth map shape: {0}".format(depth.shape))
v, u = depth.shape
depth_val = depth[v - 1, u - 1]
print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(0, depth_val))

print(dataset_handler.num_frames)

i = 30
image = dataset_handler.images[i]
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')


def extract_features(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def visualize_features(image, kp):
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)


i = 0
image = dataset_handler.images[i]
kp, des = extract_features(image)
image = dataset_handler.images_rgb[i]
visualize_features(image, kp)
print("Number of features detected in frame {0}:{1}\n".format(i, len(kp)))
print("Coordinates of the first keypoint in frame {0}:{1}".format(i, str(kp[0].pt)))


def extract_features_dataset(images, extract_features_function):
    kp_list = []
    des_list = []

    for image in images:
        kp, des = extract_features_function(image)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)

i = 0

print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

# Remember that the length of the returned by dataset_handler lists should be the same as the length of the image array
print("Length of images array: {0}".format(len(images)))


def match_features(des1, des2):
    bf = cv2.BFMatcher()
    match = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in match:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


n = 20

i = 0
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i + 1]

kp1 = kp_list[i]
kp2 = kp_list[i + 1]

des1 = des_list[i]
des2 = des_list[i + 1]

match = match_features(des1, des2)

visualize_matches(image1, kp1, image2, kp2, match[:n])

plt.show()
