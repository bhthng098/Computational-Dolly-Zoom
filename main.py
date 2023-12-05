import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float32, img_as_ubyte
from skimage.transform import warp, AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.filters import gaussian
import os
import argparse

# equation 4
def applyDigitalZoom(input, f, f_desired):
    """
    applies a digital zoom to an input image, given the image's
    current focal length and the desired output focal length

    Params:
      input:      input image
      f:          focal length of input image
      f_desired:  desired focal length of output image

    Return:
      A digitally-zoomed image with focal length f_desired
      Note: this LEAVES GRIDLINE GAPS in the output image b/c sampling mode
    """
    I = input[:,:,1]
    out_im = np.zeros(I.shape)
    im_width, im_height = I.shape

    k = f_desired / f
    u0 = np.array([im_width // 2, im_height // 2])
    center_offset = (1 - k) * u0

    for i_x in range (im_width):
        for i_y in range(im_height):
            [out_x, out_y] = k * np.array([i_x, i_y]) + center_offset
            out_x = int(out_x)
            out_y = int(out_y)
            if out_x  in range(im_width) and out_y in range(im_height):  
              out_im[out_x, out_y] = I[i_x, i_y]
    return out_im

def isPositionValid(position, width, height):
    x, y = position
    return x in range(width) and y in range(height)

def applyDigitalZoomReverse(input, f, f_desired):
    """
    applies a digital zoom to an input image, given the image's
    current focal length and the desired output focal length

    Params:
      input:      input image
      f:          focal length of input image
      f_desired:  desired focal length of output image

    Return:
      A digitally-zoomed image with focal length f_desired
      Note: this DOES NOT leaves gridline gaps in the output image b/c
            we are applying applyDigitalZoom() to the out im in reverse
    """
    I = img_as_float32(input[:,:,:])
    out_im = np.zeros(I.shape)
    im_width = I.shape[0]
    im_height = I.shape[1]

    k = f_desired / f
    u0 = np.array([im_width // 2, im_height // 2])
    center_offset = (1 - k) * u0
    print(center_offset)

    # # for each pixel in the out image
    # for o_x in range (im_width):
    #     for o_y in range(im_height):
    #         # find the corresponding input image location
    #         [in_x, in_y] = (np.array([o_x, o_y]) - center_offset) / k
    
    #         in_x = int(in_x)
    #         in_y = int(in_y)

    #         if in_x  in range(im_width) and in_y in range(im_height):
    #           # if it is in range, set the out image pixel to the input image pixel value
    #           out_im[o_x, o_y, :] = I[in_x, in_y, :]
    indices_of_out = np.indices((im_width, im_height))
    
    out_im = I[(indices_of_out - center_offset) / k] if isPositionValid((indices_of_out - center_offset) / k, im_width, im_height) else 0

    return out_im

# equation 3
def DZSynthesis(d, i_a, d_a, t):
    """
    Produces a synthesized image of the digitally-zoomed image i_a
    given the original depth map, the digitally-zoomed depth map,
    and a given t.

    All image inputs should have the same shape

    Params:
      d:    original depth map
      i_a:  digitally-zoomed image
      d_a:  digitally-zoomed depth map
      t:    desired distance between depth maps

    Return:
      The output synthesized image of i_a
    """
    I_a = img_as_float32(i_a[:,:,:])
    D = img_as_float32(d[:,:,0])
    D_a = img_as_float32(d_a[:,:,0])

    out_im = np.zeros(i_a.shape)
    im_width = i_a.shape[0]
    im_height = i_a.shape[1]

    u0 = np.array([im_width // 2, im_height // 2])
    
    D_center = D[im_width // 2, im_height // 2]

    for i_x in range(im_width):
        for i_y in range(im_height):
            d_at_x_y = D_a[i_x, i_y]
            [out_x, out_y] =  d_at_x_y * (D_center - t) / (D_center * (d_at_x_y - t)) * np.array([i_x, i_y]) + (t * (d_at_x_y - D_center) / (D_center * (d_at_x_y - t))) * u0
            out_x = int(out_x)
            out_y = int(out_y)
            # print(out_x, out_y)
            if out_x  in range(im_width) and out_y in range(im_height):  
              out_im[out_x, out_y] = I_a[i_x, i_y]
    return out_im

def show_2_images_side_by_side(im1, im2):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(im1)
    axarr[1].imshow(im2)
    plt.show()

source_file = "data/beatrice_full.jpg"
depth_file = "data/beatrice_full_depth.jpg"

I = cv2.imread(source_file)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
D = cv2.imread(depth_file)
D = cv2.cvtColor(D, cv2.COLOR_BGR2RGB)

# u_0 = I.shape[0]/2
# v_0 = I.shape[1]/2
f = 35
# K = np.array([[f, 0, u_0], [0, f, v_0], [0, 0, 1]])

output_I = applyDigitalZoomReverse(I, f, 40)
print("output I found")
output_D = applyDigitalZoomReverse(D, f, 40)
print("output D found")

cv2.imwrite(f'results/beatrice_full_35_to_40.png', cv2.cvtColor(img_as_ubyte(output_I), cv2.COLOR_RGB2BGR))
cv2.imwrite(f'results/beatrice_full_depth_35_to_40.png', cv2.cvtColor(img_as_ubyte(output_D), cv2.COLOR_RGB2BGR))

# i_digital_zoom = "results/beatrice_full.jpg"
# depth_digital_zoom = "data/beatrice_full_depth.jpg"

# I = cv2.imread(source_file)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# D = cv2.imread(depth_file)
# D = cv2.cvtColor(D, cv2.COLOR_BGR2RGB)

# i_1_dz = DZSynthesis(D, output_I, output_D, .1)
# plt.imshow(i_1_dz)
# plt.show()
# cv2.imwrite(f'results/beatrice_dz_35_to_40.png', cv2.cvtColor(img_as_ubyte(i_1_dz), cv2.COLOR_RGB2BGR))
