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
    orb = cv2.ORB_create(nfeatures=1500)
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

matches = match_features(des1, des2)

visualize_matches(image1, kp1, image2, kp2, matches[:n])


def match_features_dataset(des_list, match_features):
    matches = []
    matches = [match_features(d1, d2) for d1, d2 in zip(des_list[:-1], des_list[1:])]

    return matches


matches = match_features_dataset(des_list, match_features)


def estimate_motion(match, kp1, kp2, k, depth1=None):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    objectpoints = []

    # Iterate through the matched features
    for m in match:
        # Get the pixel coordinates of features f[k - 1] and f[k]
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        # Get the scale of features f[k - 1] from the depth map
        s = depth1[int(v1), int(u1)]

        # Check for valid scale values
        if s < 1000:
            # Transform pixel coordinates to camera coordinates using the pinhole camera model
            p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))

            # Save the results
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)

    # Convert lists to numpy arrays
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)

    # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points


i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i + 1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[i]

rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

i = 30
image1 = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    trajectory = np.zeros((3, 1))
    trajectory = [np.array([0, 0, 0])]
    T = np.eye(4)

    for i, match in enumerate(matches):
        rmat, tvec, _, _ = estimate_motion(match, kp_list[i], kp_list[i + 1], k, depth_maps[i])
        Ti = np.eye(4)
        Ti[:3, :4] = np.c_[rmat.T, -rmat.T @ tvec]
        T = T @ Ti
        trajectory.append(T[:3, 3])

    trajectory = np.array(trajectory).T

    return trajectory


depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

# Remember that the length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))

visualize_trajectory(trajectory)

plt.show()
