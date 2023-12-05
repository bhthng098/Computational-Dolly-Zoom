import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float32, img_as_ubyte
from skimage.transform import warp, AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.filters import gaussian

source_file = "data/beatrice.jpg"
depth_file = ""

I = cv2.imread(source_file)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
u_0 = I.shape[0]/2
v_0 = I.shape[1]/2


f = 35
# D = cv2.imread(depth_file)
# D = cv2.cvtColor(D, cv2.COLOR_BGR2RGB)

K = np.array([[f, 0, u_0], [0, f, v_0], [0, 0, 1]])

plt.imshow(I)
plt.show()


# plt.imshow(D)
# plt.show()

# equation 4
def applyDigitalZoom(input, f, f_desired):
    # I = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    I = input[:,:,1]
    # out_im = I
    out_im = np.zeros(I.shape)
    # out_im.fill(255)
    # print(input.shape)
    im_width, im_height = I.shape

    k = f_desired / f
    u0 = np.array([im_width // 2, im_height // 2])
    # print(u0, k)
    center_offset = (1 - k) * u0

    for i_x in range (im_width):
        for i_y in range(im_height):
            [out_x, out_y] = k * np.array([i_x, i_y]) + center_offset
            # out_x = int(np.clip(out_x, 0, im_width - 1))
            # out_y = int(np.clip(out_y, 0, im_height - 1))
            out_x = int(out_x)
            out_y = int(out_y)
            # print(out_x, out_y)
            if out_x  in range(im_width) and out_y in range(im_height):  
              out_im[out_x, out_y] = I[i_x, i_y]
    # print(np.where(out_im == 0)[3])
    return out_im


output = applyDigitalZoom(I, f, 40)
plt.imshow(output, cmap="gray")
plt.show()