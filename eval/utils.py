import numpy as np


# create a disc
def disc(r, n):

    # mask
    a = n/2
    b = n/2

    y, x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    ones = np.zeros([n, n])
    ones[mask] = 1

    return ones


# bin data
def bin_data(data, bin_size, center_x, center_y, padding_x, padding_y, normalize = True):

    inv_bin_size = int(1/bin_size)
    x_center_in_dx = int(center_x/bin_size)
    y_center_in_dy = int(center_y/bin_size)

    binned = np.zeros([inv_bin_size + 1 + padding_x + x_center_in_dx, inv_bin_size + 1 + padding_y + y_center_in_dy])
    offset_x = int(padding_x/2)
    offset_y = int(padding_y/2)

    for pair in data:

        x, y = int(np.round(inv_bin_size*pair[0])), int(np.round(inv_bin_size*pair[1]))
        #print("x:    ", x, "y:    ", y)
        #print("pair: ", pair[0], "pair: ", pair[1])
        #print(binned.shape)
        binned[x+offset_x, y+offset_y] += 1

    if (normalize):
    
        binned /= data.shape[0]

    return binned

            
