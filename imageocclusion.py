import numpy as np
from skimage.filters import gaussian
from scipy import ndimage as nd
import matplotlib.pyplot as plt

def depth_map_hole_fill(D_f, I_f):
    """
    param: 
        - fused_D: fused depth map
        - M: binary mask identifying holes

    returns: hole-filled version of the fused depth map
    """

    print("... filling in depth map!")

    M = I_f[:,:,0]
    M = np.where(M == 0, 1, 0)
    w = D_f.shape[0]
    h = D_f.shape[1]
    output = D_f

    plt.imshow(M, cmap="gray")
    plt.show()

    indices = nd.distance_transform_edt(M, return_distances=False, return_indices=True)
    output = output[tuple(indices)]

    # for x in range(w):
    #     for y in range(h):
    #         if M[x,y] == 1:
    #             # find four nearest neighbors to pixel
    #             neighbors = []
    #             for x_n in range(x-1, x+2):
    #                 if (x_n < 0 or x_n >= w): continue
    #                 for y_n in range(y-1, y+2):
    #                     if (y_n < 0 or y_n >= w): continue

    #                     #otherwise it's a valid index
    #                     neighbors.append(D_f[x_n, y_n, 0])

    #             # find neighbor with highest value
    #             if (len(neighbors) == 0): continue
    #             max_val = max(neighbors)

    #             # set output pixel to that value
    #             output[x,y,:] = max_val
    plt.imshow(output)
    plt.show()
    return output


def image_hole_filling(D_f, I_f):
    """
    params:
        - fused_I: synthesized fused image
        - fused_D: synthesized fused depth 
        - M: occulusion mask
    returns: 
        - hole-filled synthesize view I
    """

    output = I_f
    w = D_f.shape[0]
    h = D_f.shape[1]

    # get unique values from D_f
    d_u = np.sort(np.unique(D_f))
    d_u = d_u[::6]
    print("unique values: ", d_u)
    S = len(d_u)
    print(S)

    M = I_f[:,:,0]
    M = np.where(M == 0, 1, 0)
    plt.imshow(M)
    plt.show()

    M = np.stack([M, M, M], axis=-1)
    # plt.imshow(M[:,:,0])
    # plt.show()


    M_prev = np.zeros(M.shape)

    for s in reversed(range(1, S)):
        print(s)
        # depth mask Ds corresponding to depth step
        D_s = (D_f > d_u[s-1])&(D_f <= d_u[s])
        #print("D_s: ", D_s)
        I_s = I_f * D_s
        M_curr = M * D_s
        M_curr = np.logical_or(M_curr, M_prev)
        # plt.imshow(M_curr[:,:,0])
        # plt.show()

        for x in range(w):
            for y in range(h):
                if (M_curr[x,y,0] == 1 or M_curr[x,y,0] == True or M_curr[x,y,0] != False):
                    #print("in loop")
                    # get same row in I_s
                    nearest_valid_xs = []
                    for x_row in range(w):
                        if (I_s[x_row, y, 0] != 0 or I_s[x_row, y, 0] != False) and x_row != x:
                            #print("getting nearest neighbor")
                            nearest_valid_xs.append(x_row)
                    
                    # get the nearest x_row to the current x
                    nearest_x = find_nearest(nearest_valid_xs, x)
                    #print("setting to value: ", I_s[nearest_x, y, :])
                    output[x,y, :] = I_s[nearest_x, y, :]
                    M_curr[x,y] = 0
                    M[x,y] = 0
        M_prev = M_curr
    
    print("filled in holes of I_f")
    # low pass filtered on the filled-in occluded areas in output
    return (output)

def find_nearest(array, value):
    # if array is empty, return the original x location
    if (len(array) == 0): return value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]






