import cv2
import numpy as np
from digitalzoom import applyDigitalZoomReverse
from dzsynthesis import DZSynthesis, DZSynthesis_SecondForm

def image_fusion(I1, I2):
    B = I1
    B = np.where(B == 0, 1, 0)
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
