import time

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches

import files_management

image_left = files_management.read_left_image()
image_right = files_management.read_right_image()

_, image_cells = plt.subplots(1, 2, figsize=(20, 20))
image_cells[0].imshow(image_left)
image_cells[0].set_title('left image')
image_cells[1].imshow(image_right)
image_cells[1].set_title('right image')
plt.show()

plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_left)

p_left, p_right = files_management.get_projection_matrices()
np.set_printoptions(suppress=True)
print("p_left \n", p_left)
print("\np_right \n", p_right)


def compute_left_disparity_map(img_left, img_right):
    num_disparities = 6 * 16
    block_size = 11
    min_disparity = 0
    window_size = 6

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # left_matcher_BM = cv2.StereoBM_create(
    #     numDisparities=num_disparities,
    #     blockSize=block_size
    # )

    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32) / 16

    return disp_left


SGBM_time_start = time.time()
disp_left = compute_left_disparity_map(image_left, image_right)
SGBM_end_start = time.time()
print("SGBM use time: ", SGBM_end_start - SGBM_time_start)

plt.figure(figsize=(10, 10))
plt.imshow(disp_left)
plt.show()


def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = t / t[3]

    return k, r, t


k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)

print("k_left \n", k_left)
print("\nr_left \n", r_left)
print("\nt_left \n", t_left)
print("\nk_right \n", k_right)
print("\nr_right \n", r_right)
print("\nt_right \n", t_right)


def calc_depth_map(disp_left, k_left, t_left, t_right):
    f = k_left[0, 0]
    b = t_left[1] - t_right[1]

    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1

    depth_map = np.ones(disp_left.shape, np.single)
    depth_map[:] = f * b / disp_left[:]

    return depth_map


depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show()
