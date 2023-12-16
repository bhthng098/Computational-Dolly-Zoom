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
   parser.add_argument("-fo", help="original focal length in mm")
   parser.add_argument("-fmin", help="minimum focal length in mm")
   parser.add_argument("-fd", help="desired focal length in mm")
   parser.add_argument("--show", help="show intermediate images", action="store_true")
   parser.add_argument("--use_fusion_fill", help="use the fusion images to fill the holes/gaps", action="store_true")
   args = parser.parse_args()
   
   source_path = args.s
   orig_image = cv2.imread(source_path)
   orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

   depth_path = args.d
   depth_map = cv2.imread(depth_path)
   depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)

   if orig_image is None or depth_map is None:
     print("invalid filepaths")
   else:
    if args.show:
        show_2_images_side_by_side(orig_image, depth_map, "orig images")

    f_original = mmToPixels(int(args.fo)) #in pixels, not mm
    f_desired = mmToPixels(int(args.fd)) #in pixels, not mm
    
      # 1. DIGITAL ZOOM
    
    i1, d1 = generateDigitallyZoomedImageAndDepth(orig_image, depth_map, f_original, f_desired)
    if args.show:
      show_2_images_side_by_side(i1, d1, "digitally zoomed")
    
    f_min = mmToPixels(int(args.fmin))
    print(f_original, f_desired, f_min)
    t_target = f_min
    while (f_min <= t_target <= f_desired):
      t = calculateT(f_original, t_target, depth_map)

      # 2. DZ SYNTHESIS
      i_1_dz, d_1_dz = generate_I1_dz_D1_dz(depth_map, i1, d1, t)

      i_2_dz, d_2_dz = generate_I2_dz_D2_dz(depth_map, orig_image, t, f_original, f_desired)

        # IMAGE FUSION OF 1 & 2
      i_F = image_fusion(i_1_dz, i_2_dz)
      d_F = image_fusion(d_1_dz, d_2_dz)

        # HOLE FILLING DEPTH_F
      hole_filled_D = depth_map_hole_fill(d_F, i_F)

      if args.use_fusion_fill:
        hole_filled_I = image_hole_filling(hole_filled_D, i_F, i_F)
        hole_filled_I = depth_map_hole_fill(hole_filled_I, hole_filled_I)
      else:
        hole_filled_I = image_hole_filling(hole_filled_D, i_F, i1)
        hole_filled_I = depth_map_hole_fill(hole_filled_I, hole_filled_I)

      if args.show:
          plt.imshow(hole_filled_I)
          plt.show()
      
      cv2.imwrite(f'results/'+source_path.split("/", 2)[-1][:-5]+"-fillholeimage-"+str(t_target)+'.jpeg', cv2.cvtColor(img_as_ubyte(hole_filled_I), cv2.COLOR_RGB2BGR))
      t_target += 10
