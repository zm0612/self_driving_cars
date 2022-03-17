import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *

np.random.seed(1)
# np.set_printoptions(precision=2, threshold=np.nan)

dataset_handle = DatasetHandler()

image = dataset_handle.image
plt.imshow(image)

k = dataset_handle.k
print(k)

depth = dataset_handle.depth
plt.imshow(depth, cmap='jet')

segmentation = dataset_handle.segmentation
plt.imshow(segmentation)

colored_segmentation = dataset_handle.vis_segmentation(segmentation)
plt.imshow(colored_segmentation)

plt.show()
