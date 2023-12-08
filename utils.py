import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_2_images_side_by_side(im1, im2, title):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(im1)
    axarr[1].imshow(im2)
    plt.title(title)
    plt.show()
