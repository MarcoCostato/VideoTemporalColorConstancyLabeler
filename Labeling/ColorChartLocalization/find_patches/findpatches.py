import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import tqdm
import math
import skimage
import argparse
import timeit
import skimage.segmentation as seg
import sys
import os
sys.path.append(os.path.dirname(__file__))

from binarization import find_centers
from consensus import find_angles_, find_distances_, get_ordered_points, get_patches


def rotate_resize(img, mask, width):
    h,w = img.shape[:2]
    rotated = False
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        h,w = img.shape[:2]
        rotated = True

    ratio = width / w
    img = cv2.resize(img, (width, math.floor(h*ratio)))
    if mask is not None:
        mask = cv2.resize(mask, (width, math.floor(h*ratio)))
    return img, mask, rotated

def findpatches(img, mask=None, get_viz=False, t1=10, t2=200, tolerance=np.pi/12):
    img, mask, rotated = rotate_resize(img, mask, 256)
    
    points, processed, bw, edges, dist = find_centers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), mask, t1, t2) # find potential centroids of the patches
    
    # viz = np.zeros_like(mask)
    # for p in points:
    #     viz[p[0],p[1]] = 255
    # viz = edges
    # viz = bw
    # viz = dist
    # viz = processed

    # Remove points that are outside the mask
    if mask is not None: # This if is useless because if mask is not None, there are no points outside the mask. 
        points = [p for p in points if mask[p[0], p[1]] == 255]
        points = np.array(points).astype(np.int32)
    
    if len(points) < 8:
        # Launch an error (not enough points) and interrupt the execution
        raise Exception("Not enough points found")
    try:
        angle1, angle2 = find_angles_(points, tolerance=tolerance)
    except:
        raise Exception("Couldn't find orientation angles")

    dist1, dist2 = find_distances_(points, angle1, angle2, tolerance=tolerance)
    shift1 = np.array([int(dist1*math.sin(angle1)), int(dist1*math.cos(angle1))])
    shift2 = np.array([int(dist2*math.sin(angle2)), int(dist2*math.cos(angle2))])

    radius = np.round((np.min([dist1, dist2])/4)).astype(np.int32) # radius of the patch mask is 1/4 of the min(shift1, shift2)
    
    # Draw angle1 and angle2 on the image
    center = (img.shape[1]//2, img.shape[0]//2)
    end1 = center + shift1[::-1]
    end2 = center + shift2[::-1]

    # Find patches
    h,w = img.shape[:2]
    patches = get_ordered_points(points, shift1, shift2, h, w)
    if patches is None:
        raise Exception("Couldn't find a valid patch configuration")

    patches = get_patches(processed, patches)
   
    if get_viz:
        img = cv2.arrowedLine(img, center, end1, (255,0,0), 1) 
        img = cv2.arrowedLine(img, center, end2, (0,255,0), 1)
        # Draw patches on the image
        patch_mask = np.zeros_like(img)
        for i in range(len(patches)):
            p = patches[i]
            coords = p['coords']
            color = p['color']
            rgb = p['rgb']

            patch_mask = cv2.circle(patch_mask, tuple(coords[::-1]), radius, rgb, -1)

            if rotated: # rotate the coordinates back
                patches[i]["coords"] = np.array([256-coords[1], coords[0]], dtype=np.int32)

        img = cv2.addWeighted(img, 0.5, patch_mask, 0.5, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        for i in range(len(patches)):
            coords = patches[i]['coords']
            if rotated:
                patches[i]["coords"] = np.array([256-coords[1], coords[0]], dtype=np.int32)
        img = None
    # img = viz
    return patches, radius, img
    

def get_triplets(img, coords, radius):
    """
    Get the triplet of colors of the patch
    
    Args:
        img: image
        coords: coordinates of the patches
        r: radius of the patches

    Returns:
        triplets: list of triplets of colors
    """
    # Create a general mask for all the patches as a 2rx2r matrix containing a circle of radius r
    radius = int(round(radius)) // 2 * 2 + 1 # Make r the nearest odd number
    mask = cv2.circle(np.zeros((2*radius -1, 2*radius -1)), (radius-1,radius-1), radius, 255, -1).astype(np.int32)
    coords = np.array(coords).astype(np.int32)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb_triplets = []
    for i in range(len(coords)):
        padded_img = np.pad(img, ((radius,radius),(radius,radius),(0,0)), mode="constant") # Pad the image to avoid out of bounds errors
        padded_grayscale = np.pad(grayscale, ((radius,radius),(radius,radius)), mode="constant") # Pad the grayscale image to avoid out of bounds errors

        y1 = coords[i][0] + 1
        y2 = coords[i][0] + 2*radius
        x1 = coords[i][1] + 1
        x2 = coords[i][1] + 2*radius

        patch_bgr  = padded_img[y1:y2, x1:x2][mask == 255]
        patch_intensities = padded_grayscale[y1:y2, x1:x2][mask == 255]

        # Filter the colors by keeping only the ones corresponding to the intensities of the 2nd and 3rd quartile
        patch_intensities = patch_intensities.flatten()
        patch_bgr = patch_bgr[(patch_intensities >= np.quantile(patch_intensities, 0.25)) & (patch_intensities <= np.quantile(patch_intensities, 0.75))]

        b, g, r = np.mean(patch_bgr, axis=0)

        rgb_triplets += [r,g,b]
    
    return rgb_triplets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="50_8D5U5577.png")

    args = parser.parse_args()

    findpatches(args.path)