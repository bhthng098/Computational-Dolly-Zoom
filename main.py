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



def show_2_images_side_by_side(im1, im2, title):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(im1)
    axarr[1].imshow(im2)
    plt.title(title)
    plt.show()

def image_fusion(I1, I2):
    B = I1
    B = np.where(B == 0, 1, 0)

    # plt.imshow(B)
    # plt.show()  

    output = B*I2 + (1-B)*I1
    print("Images fused")
    return output

def generateDigitallyZoomedImageAndDepth(orig_image, depth_map, f_original, f_desired):
  u_0 = orig_image.shape[0]/2
  v_0 = orig_image.shape[1]/2
  K = np.array([[f_original, 0, u_0], [0, f_original, v_0], [0, 0, 1]])

  output_I = applyDigitalZoomReverse(orig_image, f_original, f_desired)
  print("digital zoom I generated")

  output_D = applyDigitalZoomReverse(depth_map, f_original, f_desired)
  print("digital zoom D generated")
  # cv2.imwrite(f'results/digzoom'+source_file_path[5:]+'.png', cv2.cvtColor(img_as_ubyte(output_I), cv2.COLOR_RGB2BGR))
  # cv2.imwrite(f'results/digzoom'+depth_file_path[5:]+'.png', cv2.cvtColor(img_as_ubyte(output_D), cv2.COLOR_RGB2BGR))
  return output_I, output_D

def generate_I1_dz_D1_dz(depth_map, i_dig_zoom, d_dig_zoom, t):
  i_1_dz = DZSynthesis(depth_map, i_dig_zoom, d_dig_zoom, t)
  print("I1 DZ generated")
  d_1_dz = DZSynthesis(depth_map, d_dig_zoom, d_dig_zoom, t)
  print("D1 DZ generated")
  
  return i_1_dz, d_1_dz

def generate_I2_dz_D2_dz(depth_map, image, t, f_original, f_desired):
  i_2_dz = DZSynthesis_SecondForm(depth_map, image, t, f_original, f_desired)
  print("I2 DZ generated")
  d_2_dz = DZSynthesis_SecondForm(depth_map, depth_map, t, f_original, f_desired)
  print("D2 DZ generated")
  return i_2_dz, d_2_dz
  # cv2.imwrite(f'results/beatrice_i_dz_35_to_40_t1e-1.png', cv2.cvtColor(img_as_ubyte(i_1_dz), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-s", help="filepath for source image")
   parser.add_argument("-d", help="filepath for depth image")
   parser.add_argument("--izoom", help="optional filepath for existing digital zoom image instead of re-generating it")
   parser.add_argument("--dzoom", help="optional filepath for existing digital zoom depth instead of re-generating it")
   parser.add_argument("--save", help="flag to save all images. default does not save them", action="store_true")
   parser.add_argument("--quiet", help="do not show images", action="store_true")
   args = parser.parse_args()
   
   source_path = args.s
   orig_image = cv2.imread(source_path)
   orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

   depth_path = args.d
   depth_map = cv2.imread(depth_path)
   depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)

   if not args.quiet:
      show_2_images_side_by_side(orig_image, depth_map, "orig images")

###### HERE ARE THE PARAMETERS WE CAN CHANGE ######
   f_original = 50
   f_desired = 85
   t = .05
###### HERE ARE THE PARAMETERS WE CAN CHANGE ######
  
   if args.izoom and args.dzoom:
      i1 = cv2.imread(args.izoom)
      i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
      d1 = cv2.imread(args.dzoom)
      d1 = cv2.cvtColor(d1, cv2.COLOR_BGR2RGB)
   else:
      i1, d1 = generateDigitallyZoomedImageAndDepth(orig_image, depth_map, f_original, f_desired)
      if not args.quiet:
        show_2_images_side_by_side(i1, d1, "digitally zoomed")
      if args.save:
        cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-zoomed-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(i1), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'results/'+depth_path.split("/", 2)[-1][:-5]+"-zoomed-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(d1), cv2.COLOR_RGB2BGR))
      
   i_1_dz, d_1_dz = generate_I1_dz_D1_dz(depth_map, i1, d1, t)
   if not args.quiet:
      show_2_images_side_by_side(i_1_dz, d_1_dz, "dolly zoom synthesized images 1")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-1dz-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(i_1_dz), cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'results/'+depth_path.split("/", 2)[-1][:-5]+"-1dz-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(d_1_dz), cv2.COLOR_RGB2BGR))
  
   i_2_dz, d_2_dz = generate_I2_dz_D2_dz(depth_map, orig_image, t, f_original, f_desired)
   if not args.quiet:
      show_2_images_side_by_side(i_2_dz, d_2_dz, "dolly zoom synthesized images 2")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-2dz-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(i_2_dz), cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'results/'+depth_path.split("/", 2)[-1][:-5]+"-2dz-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(d_2_dz), cv2.COLOR_RGB2BGR))
  
   i_F = image_fusion(i_1_dz, i_2_dz)
   d_F = image_fusion(d_1_dz, d_2_dz)
   if not args.quiet:
      show_2_images_side_by_side(i_F, d_F, "images fused")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-fused-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(i_F), cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'results/'+depth_path.split("/", 2)[-1][:-5]+"-fused-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(d_F), cv2.COLOR_RGB2BGR))

