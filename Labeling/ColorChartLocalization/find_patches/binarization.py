import numpy as np
import cv2
import os
import skimage
from scipy.ndimage import convolve1d

import matplotlib.pyplot as plt
import ipdb


def find_centers(img, mask=None, t1=10, t2=200):
    """This functions finds the centers of square-shaped objects in a grayscale image"""
    
    if mask is not None:
        img = np.where(mask==0, 0, img)

    img = cv2.equalizeHist(img) # Equalize histogram
    img = (skimage.filters.unsharp_mask(img, radius=5, amount=2)*255).astype(np.uint8) # Sharpen image with unsharp mask
    img = cv2.bilateralFilter(img, 9, 200, 200)

    # t1,t2 = thresh_selection_matlab(img)
    # t1, t2 = thresh_selection_otsu(img)
    edges = cv2.Canny(img, t1, t2) # Find edges
    # edges, thresh = canny_edge_detection(img)

    dist = cv2.distanceTransform(255- edges, cv2.DIST_L2, 5) # Find distance transform
    if mask is not None:
        dist = np.where(mask==0, 0, dist)
    
    bw = np.where(dist>1 , 255, 0).astype(np.uint8) # Threshold distance transform
    # Erode
    bw = cv2.erode(bw, np.ones((3,3), np.uint8), iterations=3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8) # Find connected components

    #Filter components:
    # 1. Remove components that are too small
    # 2. Remove components that are too large
    # 3. Remove components that have centroids outside of the component
    valid_centroids = []
    for i in range(num_labels):
        component = np.where(labels==i, 255, 0).astype(np.uint8)
        centroid = centroids[i].astype(np.int32)
        if stats[i, cv2.CC_STAT_AREA] < 100 or stats[i, cv2.CC_STAT_AREA] > 1500 or component[centroid[1], centroid[0]] == 0:
            labels[labels==i] = 0
        else:
            valid_centroids.append(centroid)
    points = np.array(valid_centroids)

    bw = np.where(labels>0, 255, 0).astype(np.uint8) # Keep only valid components

    points = points[:,::-1]

    return points, img, bw, edges, dist


def thresh_selection_otsu(image):
    thresh, _ = cv2.threshold(image, 0,255,cv2.THRESH_OTSU)

    return int(0.5*thresh), int(thresh)


def thresh_selection_matlab(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    dx,dy = smoothGradient(image, np.sqrt(2))

    magGrad = np.hypot(dx, dy)
    magmax = np.max(magGrad)
    if magmax > 0:
        magGrad = magGrad/magmax

    m,n = magGrad.shape

    counts, bin_edges = np.histogram(magGrad, bins=64, range=(0, 1)) # Compute the histogram of magGrad
    cumulative_counts = np.cumsum(counts) # Compute the cumulative sum of the histogram
    high_thresh_bin = np.argmax(cumulative_counts > 0.7 * m * n)

    # Convert the bin index to a threshold value
    highThresh = bin_edges[high_thresh_bin]

    # Calculate the low threshold
    lowThresh = 0.4 * highThresh

    return int(lowThresh*255), int(highThresh*255)

def smoothGradient(I, sigma):
    """
    Compute the smoothed numerical gradient of the image I using a Derivative of Gaussian filter.

    Parameters:
    - I: Input grayscale image (2D numpy array).
    - sigma: Standard deviation of the Gaussian filter.

    Returns:
    - GX: Gradient of the smoothed image along the x-axis.
    - GY: Gradient of the smoothed image along the y-axis.
    """
    # Determine filter length
    filter_extent = int(np.ceil(4 * sigma))
    x = np.arange(-filter_extent, filter_extent + 1)

    # Create 1-D Gaussian Kernel
    c = 1 / (np.sqrt(2 * np.pi) * sigma)
    gaussKernel = c * np.exp(-(x**2) / (2 * sigma**2))

    # Normalize to ensure the kernel sums to one
    gaussKernel /= np.sum(gaussKernel)

    # Create 1-D Derivative of Gaussian Kernel
    derivGaussKernel = np.gradient(gaussKernel)

    # Normalize to ensure the kernel sums to zero
    neg_vals = derivGaussKernel < 0
    pos_vals = derivGaussKernel > 0
    derivGaussKernel[pos_vals] /= np.sum(derivGaussKernel[pos_vals])
    derivGaussKernel[neg_vals] /= -np.sum(derivGaussKernel[neg_vals])

    # Compute the smoothed gradient along the x-axis (horizontal)
    GX = convolve1d(I, gaussKernel, axis=1, mode='reflect')  # Smooth in x-direction
    GX = convolve1d(GX, derivGaussKernel, axis=0, mode='reflect')  # Gradient in y-direction

    # Compute the smoothed gradient along the y-axis (vertical)
    GY = convolve1d(I, gaussKernel, axis=0, mode='reflect')  # Smooth in y-direction
    GY = convolve1d(GY, derivGaussKernel, axis=1, mode='reflect')  # Gradient in x-direction

    return GX, GY