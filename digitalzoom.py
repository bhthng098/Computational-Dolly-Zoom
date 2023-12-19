import cv2
import numpy as np
from skimage import img_as_float32, img_as_ubyte

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
    I = img_as_float32(input[:,:,:])
    out_im = np.zeros(I.shape)
    im_width = I.shape[0]
    im_height = I.shape[1]

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

    return applyDigitalZoom(input, f, f_desired)
    I = img_as_float32(input[:,:,:])
    out_im = np.zeros(I.shape)
    im_width = I.shape[0]
    im_height = I.shape[1]

    k = f_desired / f
    u0 = np.array([im_width // 2, im_height // 2])
    center_offset = (1 - k) * u0

    # for each pixel in the out image
    for o_x in range (im_width):
        for o_y in range(im_height):
            # find the corresponding input image location
            [in_x, in_y] = (np.array([o_x, o_y]) - center_offset) / k
    
            in_x = int(in_x)
            in_y = int(in_y)

            if in_x  in range(im_width) and in_y in range(im_height):
              # if it is in range, set the out image pixel to the input image pixel value
              out_im[o_x, o_y, :] = I[in_x, in_y, :]

    # AN ATTEMPT TO DO WITHOUT FOR LOOPS. FAILED. DO NOT RUN, WILL TAKE FOREVER
    # indices_of_out = np.where(out_im[:,:,0] == out_im[:,:,0])
    # positions_of_out = np.dstack(indices_of_out)[0]
    
    # all_possible_positions = (positions_of_out - center_offset) / k
    # all_possible_positions = all_possible_positions.astype(int)
    # valid_positions = all_possible_positions[ (0<=all_possible_positions[:,1]) & (all_possible_positions[:,1]<im_height) ]
    # valid_positions = valid_positions[ (0<=valid_positions[:,0]) & (valid_positions[:,0]<im_width) ]

    # out_im[valid_positions, :] = I[valid_positions, :]

    return out_im
