import glob

import numpy as np
from matplotlib import pyplot as plt
import cv2
import pdb
from scipy import ndimage

def fspecial_gauss(size, sigma):
    # https: // stackoverflow.com / questions / 17190649 / how - to - obtain - a - gaussian - filter - in -python
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])

    return (rows, cols)

def blobs():
    # blobs: blob detection using Laplacian pyramid
    im = cv2.imread('butterfly3.png')
    orig_im = im
    if (im.shape[2] > 1):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)  # to float32

    sigma = 2 #

    # plt.figure()
    # plt.imshow(im, cmap=plt.get_cmap('gray'))
    # plt.title('original')
    # plt.show()

    # consider t keypoints at each level
    t = 100
    keypoints = np.zeros((0, 3))

    for i in range(1, 8+1):
        # filter, compute difference between two Gaussian levels
        fil = fspecial_gauss(np.ceil(3 * sigma), sigma)
        im_fil = ndimage.convolve(im, fil, mode='constant')
        blobs_at_this_level = im - im_fil

        # find largest t values and show them on the image
        inds = np.argsort(blobs_at_this_level.ravel())[::-1] # descending sort
        [rows, cols] = ind2sub(im.shape, inds[t:])
        blobs_at_this_level[ rows, cols ] = 0 # hide all pixels except those with the top t responses

        plt.figure()
        plt.imshow(blobs_at_this_level, cmap=plt.get_cmap('gray'))
        plt.title('sigma=' + str(sigma ** i))
        plt.show()

        # get x / y / scale for all keypoints at this level
        [rows, cols] = ind2sub(im.shape, inds[0:t])
        rep_mat = np.tile(sigma ** i, (t, 1))
        rows = rows.reshape((len(rows), 1))
        cols = cols.reshape((len(cols), 1))
        keypoints = np.vstack((keypoints, np.hstack((rows, cols, rep_mat))))

        # overwrite image with blurred version
        im = im_fil

    # skipped: non - max suppression!

    # show all keypoints
    plt.figure()
    plt.imshow(orig_im)

    # hold on
    for i in range(keypoints.shape[0]):
        rad = keypoints[i][2] ** (1/3)
        plt.plot(keypoints[i][1], keypoints[i][0], marker="o", markersize=rad, markeredgecolor="red", markerfacecolor="green")
    plt.show()

def autocorr_surface(r, c):
    ones_mat = np.ones((5, 15))
    ones_mat2 = np.ones((10, 6))
    zeros_mat = np.zeros((10, 9))
    im = np.vstack( (ones_mat, np.hstack((ones_mat2, zeros_mat)) ) )

    # plot images
    plt.figure()
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

    # to define what x, y range over
    num_neighbors_away = 1

    # to define what u, v range over
    max_offset = 2

    # initialize autocorrelation surface
    E = np.zeros((max_offset * 2 + 1, max_offset * 2 + 1))

    for u in range(-max_offset, max_offset+1):
        for v in range(-max_offset, max_offset+1):
            # using loops
            S = 0
            for x in range(r-num_neighbors_away, r + num_neighbors_away+1):
                for y in range(c-num_neighbors_away, c + num_neighbors_away+1):
                   S = S + (im[x, y] - im[x + u, y + v])**2

            # without loops
            A = im[r - num_neighbors_away:r + num_neighbors_away+1, c - num_neighbors_away: c + num_neighbors_away+1]
            B = im[r - num_neighbors_away + u:r + num_neighbors_away + u+1, c - num_neighbors_away + v: c + num_neighbors_away + v+1]

            S2 = sum(sum((A - B)**2))
            assert (S == S2)

            # save as error surface entry
            E[max_offset + u, max_offset + v] = S

    # plot autocorrelation surface
    plt.figure()
    plt.imshow(E, cmap=plt.get_cmap('gray'))
    plt.show()

# Main Function
autocorr_surface(4, 4)
# autocorr_surface(7, 7)
# autocorr_surface(10, 10)

# blobs() # blobs