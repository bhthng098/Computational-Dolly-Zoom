import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_2_images_side_by_side(im1, im2, title):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(im1)
    axarr[1].imshow(im2)
    plt.title(title)
    plt.show()

def calculateT(f_original, f_desired, D):
    width = D.shape[0]
    height = D.shape[1]
    fov1 = calculateFOV(f_original, width)
    fovDZ = calculateFOV(f_desired, width)
    print("Fov1: ",  fov1)
    print("FovDZ: ", fovDZ)

    D_0 = np.array([width // 2, height // 2])

    t = D_0 * (np.tan(fovDZ/2) - np.tan(fov1/2)) / (np.tan(fovDZ / 2))

    print("t: ", t)
    return t

def calculateFOV(f, img_width):
    return 2*np.arctan((img_width/2)/f)

def calculateF(FOV, img_width):
    f = (img_width/2) / np.tan(FOV/2)
    return f

def mmToPixels(mm):
    return int(mm * 3.779)