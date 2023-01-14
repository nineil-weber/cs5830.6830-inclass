import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb
from scipy import ndimage # conda install scipy==1.9.3

def read_sample_im():
    im_name = './../Module_1_python/pittsburgh.png'
    im = cv2.imread(im_name)  # Read a PNG image

    resize_percent = 0.5
    width = int(im.shape[1] * resize_percent)
    height = int(im.shape[0] * resize_percent)
    new_dim = (width, height)
    im = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)
    return im

def filters_slide16():
    # Filters: Slide 16
    im = read_sample_im()

    print('data type: ', im.dtype)
    print('min: ', im.min())
    print('max: ', im.max())
    im2 = im.astype(np.float32)  # Convert to float 32, so we can do floating operations

    # add noise
    sigma = 100
    rand_matrix = np.random.rand(*im2.shape)
    noise = rand_matrix * sigma
    im_noise = im2 + noise
    im_noise = np.where(im_noise < 0, 0, im_noise)
    im_noise = np.where(im_noise > 255, 255, im_noise)
    im_noise = im_noise.astype(np.uint8)

    cv2.imshow('image', im)  # Display the image
    cv2.waitKey(0)
    cv2.imshow('Image + noise', im_noise)
    cv2.waitKey(0)

    # Remove filter
    clean_im = ndimage.median_filter(im, 5)
    cv2.imshow('Image + noise', im_noise)
    cv2.waitKey(0)

def fspecial_gauss(size, sigma):
    # https: // stackoverflow.com / questions / 17190649 / how - to - obtain - a - gaussian - filter - in -python
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def filters_slide26():
    im = read_sample_im()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # To gray scale
    hsize = 10
    sigma = 5
    filter = fspecial_gauss(hsize, sigma)

    # Filter visualization
    fig = plt.figure()
    cax = plt.matshow(filter)
    fig.colorbar(cax)
    plt.show()

    filt_im = ndimage.convolve(im, filter, mode='constant')
    pdb.set_trace()

    cv2.imshow('image', im)  # Display the image
    cv2.waitKey(0)
    cv2.imshow('Image filtered', filt_im)
    cv2.waitKey(0)

def examples():
    from scipy import ndimage
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    im = read_sample_im()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # To gray scale

    # result = ndimage.sobel(im)
    result = ndimage.gaussian_filter(im, 5)
    # result = ndimage.laplace(im)

    # print( result.min() )
    # print(result.max())

    ax1.imshow(im)
    ax2.imshow(result)
    plt.show()


# main
filters_slide16()
# filters_slide26()
# examples()
