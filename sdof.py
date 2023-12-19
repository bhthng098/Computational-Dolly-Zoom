import numpy as np
from skimage.filters import gaussian
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float32, img_as_ubyte
from utils import Laplacian_Pyramid_Blending_with_mask

def sdof(I, D, t):
  """
  Depth-aware blurring that involves a "circle of confusion"

  """
  # D = D[:,:,0]
  d_width, d_height, _ = D.shape
  D_0 = D[d_width // 2 + 50, d_height // 2 + 50, 0]
  print(D.shape)
  # output = I.copy()
  I = img_as_float32(I)
  output = I

  
  A = 1.8
  m = 1
  
  depths = np.sort(np.unique(D))
  depths = depths[::30]
  print(depths)
  # print(len(depths))

  # for s in reversed(range(1, len(depths))):
  #   depth = depths[s]
  #   dist = np.abs(1 - np.abs(depths[s] - D_0) / (depths[-1])) * 10
  #   print(dist)
  #   # print(np.abs(int(depth) - int(D_0)))
  #   blurred_output = gaussian(I, dist, channel_axis=-1)
  #   D_s = (D > depths[s-1])&(D <= depths[s])
  #   # print(D_s, np.any(D_s))

  #   output[:,:,0] = (np.where(D_s[:,:,0] == False, output[:,:,0], blurred_output[:,:,0]))
  #   output[:,:,1] = (np.where(D_s[:,:,0] == False, output[:,:,1], blurred_output[:,:,1]))
  #   output[:,:,2] = (np.where(D_s[:,:,0] == False, output[:,:,2], blurred_output[:,:,2]))

  #   plt.imshow(output)
  #   plt.show()
  
  blurred = gaussian(I, 10, channel_axis=-1)
  # foreground_mask = np.where((D > (D_0 - 40))&(D <= D_0 + 40) == True, 0, 1)
  foreground_mask = np.zeros((d_width, d_height))
  foreground_mask[D[:,:,0] < D_0 - 80] = 1
  foreground_mask[D[:,:,0] > D_0 + 80] = 1
  foreground_mask = np.stack((foreground_mask, foreground_mask, foreground_mask), axis=-1)

  output = Laplacian_Pyramid_Blending_with_mask(blurred, I, img_as_float32(foreground_mask), num_levels=3)
  # output[foreground_mask == 1] = blurred[foreground_mask == 1]
  plt.imshow(foreground_mask, cmap="gray")
  plt.show()
  plt.imshow(output)
  plt.show()
  return output


filename = "./results/dog-fillholeimage-50-80.jpeg"
I = cv2.imread(filename)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

filename = "./results/dog-fillhole-50-80.jpeg"
depth = cv2.imread(filename)
depth = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

output = sdof(I, depth, (-230.5, -173))
cv2.imwrite("out-test.jpeg", cv2.cvtColor(img_as_ubyte(output), cv2.COLOR_RGB2BGR))
# plt.imshow(output)
# plt.show()