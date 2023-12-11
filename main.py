import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float32, img_as_ubyte
import os
import argparse
from generateImages import *
from utils import *
from imageocclusion import *

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
   f_original = 26
   f_desired = 40
   t = .07
###### HERE ARE THE PARAMETERS WE CAN CHANGE ######
  
    # 1. DIGITAL ZOOM
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
      
    # 2. DZ SYNTHESIS
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
  
    # IMAGE FUSION OF 1 & 2
   i_F = image_fusion(i_1_dz, i_2_dz)
   d_F = image_fusion(d_1_dz, d_2_dz)

   if not args.quiet:
      show_2_images_side_by_side(i_F, d_F, "images fused")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-fused-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(i_F), cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'results/'+depth_path.split("/", 2)[-1][:-5]+"-fused-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(d_F), cv2.COLOR_RGB2BGR))

    # HOLE FILLING DEPTH_F
   hole_filled_D = depth_map_hole_fill(d_F, i_F)

   if not args.quiet:
      show_2_images_side_by_side(hole_filled_D, d_F, "hole filled depth map")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-fillhole-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(hole_filled_D), cv2.COLOR_RGB2BGR))

   plt.imshow(i_F)
   plt.show()
   hole_filled_I = image_hole_filling(hole_filled_D, i_F)
   if not args.quiet:
      show_2_images_side_by_side(hole_filled_I, i_F, "hole filled imag3")
   if args.save:
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-fillholeimage-"+str(f_original)+"-"+str(f_desired)+'.jpeg', cv2.cvtColor(img_as_ubyte(hole_filled_I), cv2.COLOR_RGB2BGR))

