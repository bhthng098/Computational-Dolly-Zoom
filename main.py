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
def applyDigitalZoom(input, f, f_desired, u_0, v_0):
    k = f/f_desired
    output = k*input + (1-k)*(u_0, v_0)
    return output

output = applyDigitalZoom(I, f, 40, u_0, v_0)
plt.imshow(output)
plt.show()