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
    d_u = np.unique(D_f)
    S = len(d_u)

    M = I_f[:,:,0]
    M = np.where(M == 0, 1, 0)

    M_prev = np.zeros(M.shape)

    for s in reversed(range(1, S)):
        # depth mask Ds corresponding to depth step
        D_s = (D_f > d_u[s-1])&(D_f <= d_u[s])
        I_s = I_f * D_s
        M_curr = M * D_s
        M_curr = M_curr | M_prev

        for x in range(w):
            for y in range(h):
                if (M_curr[x,y] == 1):
                    nearest_valid = I_s[x, y, :] ## Fix!!!!
                    output[x,y, :] = nearest_valid
                    M_curr[x,y] = 0
                    M[x,y] = 0
        M_prev = M_curr
    
    print("filled in holes of I_f")
    # low pass filtered on the filled-in occluded areas in output
    return gaussian(output)








