import cv2
import numpy as np
from skimage import img_as_float32, img_as_ubyte

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
            if out_x in range(im_width) and out_y in range(im_height):  
              out_im[out_x, out_y] = I_a[i_x, i_y]
    return out_im

# equation 5
def DZSynthesis_SecondForm(d, i, t, f_original, f_desired):
    """
    Produces a synthesized image of the original image I.
    Given the original depth map, original focal length,
    desired focal length, and a given t.

    All image inputs should have the same shape

    Params:
      d:          original depth map
      i:          original image
      t:          desired distance between depth maps
      f_original: original focal length
      f_desired:  desired focal length

    Return:
      The output synthesized image I2_DZ from the original image
    """
    I = img_as_float32(i[:,:,:])
    D = img_as_float32(d[:,:,0])

    out_im = np.zeros(I.shape)
    im_width = I.shape[0]
    im_height = I.shape[1]

    u0 = np.array([im_width // 2, im_height // 2])
    D_center = D[im_width // 2, im_height // 2]

    k = (D_center - t)/D_center
    k_ = f_original / f_desired

    # for i_x in range(im_width):
    #     for i_y in range(im_height):
    #         d_at_x_y = D[i_x, i_y]
    #         [out_x, out_y] =  ((d_at_x_y * k / ((d_at_x_y - t) * k_))*(np.array([i_x, i_y])-u0)) + u0
    #         out_x = int(out_x)
    #         out_y = int(out_y)
    #         # print(out_x, out_y)
    #         if out_x  in range(im_width) and out_y in range(im_height):  
    #           out_im[out_x, out_y] = I[i_x, i_y]

    # for each pixel in the out image
    for o_x in range (im_width):
        for o_y in range(im_height):
            d_at_x_y = D[o_x, o_y]

            # find the corresponding input image location
            [in_x, in_y] =  (( ((d_at_x_y - t) * k_) / (d_at_x_y * k))*(np.array([o_x, o_y]) - u0)) + u0
    
            in_x = int(in_x)
            in_y = int(in_y)

            if in_x  in range(im_width) and in_y in range(im_height):
              # if it is in range, set the out image pixel to the input image pixel value
              out_im[o_x, o_y, :] = I[in_x, in_y, :]

    return out_im
