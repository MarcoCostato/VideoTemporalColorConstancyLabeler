import numpy as np
import cv2
import os
import skimage
import ipdb
import math
import matplotlib.pyplot as plt



"""Here are defined all the functions thate make estimations by using a consensus approach."""
 
def init_shifts_matrices():
    matrices= []
    for i in range(4):
        for j in range(6):
            
             # Define matrix (24x6x2) with (0,0) point at i,j 
            mat = np.zeros((4,6,2))
            for k in range(4):
                for l in range(6):
                    mat[k,l] = np.array([k-i, l-j])

            matrices.append(mat)
    
    return np.array(matrices) 

SHIFT_MATRICES = init_shifts_matrices()

def find_angles(points, tolerance = np.pi/18, img = None):
    """Find the two most common angles between the points in the list points.
    A tolerance is used to consider two angles equal.
    Args:
        points: list of points (row, col)
        tolerance: tolerance in radians to consider two angles equal
    
    Returns:
        top2angles: list of the two most common angles
    """

    angles = []
    # Compute all possible vectors between points
    for i in range(len(points)):
        vectors = []
        for j in range(len(points)):
            if i != j:
                vector = points[i] - points[j]
                vectors.append(vector)
        vectors = np.array(vectors)

        vectors = vectors[np.argsort(np.linalg.norm(vectors, axis=1))] # sort by length
        vectors = vectors[:2] # keep only the two shortest vectors

        angle1 = math.atan2(vectors[0][0], vectors[0][1])
        angle2 = math.atan2(vectors[1][0], vectors[1][1])
        
        if img is not None:
            # Draw the angles on the image
            start = tuple(points[i][::-1])
            end1 = tuple((points[i] - vectors[0])[::-1])
            end2 = tuple((points[i] - vectors[1])[::-1])
            cv2.arrowedLine(img, start, end1, (0,0,255), 1)
            cv2.arrowedLine(img, start, end2, (0,0,255), 1)

            

        angles.append(angle1)
        angles.append(angle2)

    
    angles = np.array(angles)

    angles[angles<0] = angles[angles<0] + np.pi # map angles to [0, pi]


    # Split the angles into clusters using tolerance
    angles = np.sort(angles)
    clusters = []
    for i in range(len(angles)):
        if i == 0:
            new_cluster = [angles[i]]
        elif angles[i] - angles[i-1] <= tolerance:
                new_cluster.append(angles[i])
        else:
            clusters.append(new_cluster)
            new_cluster = [angles[i]]
    clusters.append(new_cluster)  

    # Choose two biggest clusters
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    cluster1 = clusters[0]
    cluster2 = clusters[1]

    # Compute the mean angle of each cluster
    mean_angle1 = np.mean(cluster1)
    mean_angle2 = np.mean(cluster2)

    # Return the two most common angles
    return mean_angle1, mean_angle2

def find_angles_(points, tolerance = np.pi/18):
    """
    New version of the find_angles function that uses matrix operations (numpy) instead of for loops for better performance.
    """

    vectors = points[:, None, :] - points[None, :, :] # Compute all possible vectors between points
    vectors = vectors[np.arange(vectors.shape[0])[:, None], np.argsort(np.linalg.norm(vectors, axis=2)), :]  # Sort each row separatly by np.linalg.norm (length of the vector)
    vectors = vectors[:, 1:3, :] # Keep only the two shortest vectors (the first one is the null vector)

    angles = np.arctan2(vectors[:, :, 0], vectors[:, :, 1]) # Compute the angles of the vectors
    angles = angles.reshape(-1) # Flatten the array

    angles[angles<0] = angles[angles<0] + np.pi # map angles to [0, pi]

    # Split the angles into clusters using tolerance
    angles = np.sort(angles)
    clusters = split_tolerance(angles, tolerance) # Split the angles into clusters using tolerance
    
    # Choose two biggest clusters
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    # Select the median angles from the two biggest clusters
    median_angle0 = np.median(clusters[0])
    median_angle1 = np.median(clusters[1])

    # Return the two most common angles
    return median_angle0, median_angle1
    
def split_tolerance(arr, tolerance):
    """Function that splits the array arr into clusters using the tolerance.Expects a sorted array."""
    splits = []
    for i in range(len(arr)):
        if i == 0 or arr[i] - arr[i-1] > tolerance:
            splits.append([arr[i]])
        else: 
            splits[-1].append(arr[i])
    # Check if the first and last cluster are in the same tolerance range
    if (np.mean(splits[0]) - np.mean(splits[-1]) +np.pi) <= tolerance:
        splits[0].extend(np.add(splits[-1], -np.pi))
        splits.pop()
        splits[0] = np.add(splits[0], np.pi) if np.mean(splits[0]) < 0 else splits[0] # remap angles to [0, pi]
    
    return splits

def find_distances(points, angle1, angle2, tolerance=np.pi/18):
    dist1 = []
    dist2 = []

    for i in range(len(points)):
        d1 = 0
        d2 = 0
        for j in range(len(points)):
            if i!=j:
                angle = math.atan2(points[i][0] - points[j][0], points[i][1] - points[j][1])
                angle = angle + np.pi if angle < 0 else angle
                if abs(angle-angle1) < tolerance:
                    candidate = abs(np.linalg.norm(points[i] - points[j]))
                    if d1 == 0 or candidate < d1:
                        d1 = candidate
                elif abs(angle-angle2) < tolerance:
                    candidate = abs(np.linalg.norm(points[i] - points[j]))
                    if d2 == 0 or candidate < d2:
                        d2 = candidate
        
        dist1.append(d1)
        dist2.append(d2)

    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    
    # Remove the zeros
    dist1 = dist1[dist1!=0]
    dist2 = dist2[dist2!=0]


    return np.median(dist1), np.median(dist2)

def find_distances_(points, angle1, angle2, tolerance=np.pi/18):
    """
    New version of the find_distances function that uses matrix operations (numpy) instead of for loops for better performance.
    """
    vectors = points[:, None, :] - points[None, :, :] # Compute all possible vectors between points
    angles = np.arctan2(vectors[:, :, 0], vectors[:, :, 1]) # Compute the angles of the vectors
    
    angles[angles<0] = angles[angles<0] + np.pi # map angles to [0, pi]


 
    angles1 = np.where(np.logical_or(np.abs(angles - angle1) < tolerance, np.abs(np.abs(angles - angle1) - np.pi) < tolerance), angles, np.nan) # find angles that are in the tolerance range
    angles2 = np.where(np.logical_or(np.abs(angles - angle2) < tolerance, np.abs(np.abs(angles - angle2) - np.pi) < tolerance), angles, np.nan) # find angles that are in the tolerance range


    distances1 = np.where(np.isnan(angles1), np.nan, np.linalg.norm(vectors, axis=2)) # Compute the distances of the vectors
    distances2 = np.where(np.isnan(angles2), np.nan, np.linalg.norm(vectors, axis=2)) # Compute the distances of the vectors

    # Remove zeros 
    distances1 = np.where(distances1==0, np.nan, distances1)
    distances2 = np.where(distances2==0, np.nan, distances2)
    
    # sort the distances and remove the zeros
    distances1 = np.sort(distances1)
    distances2 = np.sort(distances2)

    # Keep only the first distance of each row
    distances1 = distances1[:, 0]
    distances2 = distances2[:, 0]

    distances1 = distances1[~np.isnan(distances1)]
    distances2 = distances2[~np.isnan(distances2)]
    

    d1 = np.median(distances1)
    d2 = np.median(distances2)

    return d1, d2

def get_ordered_points(points, shift1, shift2, h, w):
    """
    Function that returns a matrix of shape (24,6,2) or containing the ordered points of the colorchecker.
    """

    hypotheses = np.concatenate((find_grids(points, shift1, shift2), find_grids(points, shift2, shift1)), axis=0)
   
    # Filter out the hypotheses that are not valid (i.e. the grids with negative coordinates or coordinates that are out of the image bounds)
    filtered_hypotheses = filter_hypotheses(hypotheses, h, w)

    if filtered_hypotheses.shape[0] == 0:
        return None

    # Compute the score of each hypothesis
    tolerance_radius = max(np.linalg.norm(shift1), np.linalg.norm(shift2))//4 # The tolerance radius is set as 1/4 of the length of the longest shift vector
    scores = compute_scores(filtered_hypotheses, points, tolerance_radius)    
    # Filter out the hypotheses that have a score lower than max_score
    max_score = np.max(scores)

    filtered_hypotheses = filtered_hypotheses[scores == max_score]

    # Clusterize the hypotheses with tolerance_radius
    final_grid = merge_grids(filtered_hypotheses, tolerance_radius)

    return final_grid

def find_grids(points, shift2, shift1):
    """
    Function that computes all possible grids for each point in points.
    The result is a matrix of shape (n, 24, 4, 6, 2) where n is the number of points.
    For each of the n points, 24 possible positions (and therefore 24 different 4x6 grids of coordinates) are computed.
    """
    shift_matrices = SHIFT_MATRICES.copy() # The shift matrices are the same for each point, so we can compute them once and then use them for each point
    
    # Make a matrix of shape (4, 6, 2) containing the shift1 vector in each element
    shift1_matrix = np.repeat([shift1], 6*4*24, axis=0).reshape(24,4,6,2)
    # Make a matrix of shape (4, 6, 2) containing the shift2 vector in each element
    shift2_matrix = np.repeat([shift2], 6*4*24, axis=0).reshape(24,4,6,2)
    # Compute the actual shift matrices using the shift1 and shift2 matrices
    shift_matrices = np.multiply(np.stack([shift_matrices[:,:,:,0], shift_matrices[:,:,:,0]], axis=3), shift1_matrix) + np.multiply(np.stack([shift_matrices[:,:,:,1], shift_matrices[:,:,:,1]], axis=3), shift2_matrix)

    
    # Make a matrix of shape (n, 24, 4, 6, 2) containing the sum of the n points and the shift_matrices
    points_matrix = np.repeat(points, 24*4*6, axis=0).reshape(len(points), 24, 4, 6, 2)
    grids = points_matrix + shift_matrices

    return grids

def filter_hypotheses(hypotheses, h, w):
    """
    This function filters out the hypotheses that are not valid, i.e. the grids with negative coordinates or coordinates that are out of the image bounds.
    
    Args:
        hypotheses: matrix of shape (n, 24, 4, 6, 2) containing the hypotheses for each point
        h: height of the image
        w: width of the image
    """

    # Compute the min_matrix, with shape (n, 24, 2) containing the min values for each grid
    min_matrix = np.min(hypotheses, axis=(2,3))
    # Compute the max_matrix, with shape (n, 24, 2) containing the max values for each grid
    max_matrix = np.max(hypotheses, axis=(2,3))

    #############################################################################################################################
    # If the min_matrix or the max_matrix have negative values, then the grid is not valid.                                     #                                                                                                   #
    # If the min_matrix or the max_matrix have values that are greater than the image size, then the grid is not valid.         #
    # Therefore, we can filter out the grids that are not valid by checking if the min_matrix or the max_matrix have negative   #
    # values or values that are greater than the image size.                                                                    #
    # Compute a mask of shape (n, 24) containing True if the grid is valid and False otherwise                                  #
    #############################################################################################################################
    scores = np.where(min_matrix[:,:,0]>=0, True, False) & np.where(min_matrix[:,:,1]>=0, True, False) & np.where(max_matrix[:,:,0]<h, True, False) & np.where(max_matrix[:,:,1]<w, True, False)

    return hypotheses[scores]

def compute_scores(grids, points, tolerance_radius):
    """
    Function that evaluates the grids based on the points.
    """
    grids = grids.reshape(grids.shape[0]*grids.shape[1], grids.shape[2], grids.shape[3], grids.shape[4]) if len(grids.shape) == 5 else grids # reshape the grids to compact the first two dimensions

    # For each grid (n_grids x 4 x 6 x 2) and each point (n_points x 2), compute the distances matrix of shape (n_grids x 4 x 6 x n_points)
    distances = np.linalg.norm(grids.reshape(grids.shape[0], grids.shape[1], grids.shape[2], 1, grids.shape[3]) - points.reshape(1, 1, 1, points.shape[0], points.shape[1]), axis=4)

    # Compute the min distance for each point in each grid (n_grids x 4 x 6)
    min_distances = np.min(distances, axis=3)

    # Compute the score for each grid: the score is the number of points that are within the tolerance radius
    scores = np.where(min_distances <= tolerance_radius, 1, 0)
    scores = np.sum(scores, axis=(1,2))
    return scores

def merge_grids(grids, tolerance_radius):
    """
    Grids is a matrix of shape (n_grids, 4, 6, 2). This function returns a list of clusters, separating the grids that have an L2 distance bigger that the tolerance_radius
    """
    splits = []
    for i in range(len(grids)):
        if i == 0 or np.mean(np.linalg.norm(grids[i] - grids[i-1], axis=2)) > tolerance_radius:
            splits.append([grids[i]])
        else:
            splits[-1].append(grids[i])
    
    # Compute the pointwise mean of the grids in the biggest cluster
    biggest_cluster = max(splits, key=len)
    final_grid = np.mean(biggest_cluster, axis=0)
    final_grid = np.array(final_grid, dtype=np.int32)
    return final_grid

def get_patches(img, points):
    """
    This function uses the points coordinates and the image to find the correct orientation of the points matrix and associate the correct color to each point.
    """
    img = cv2.GaussianBlur(img, (9,9), 0)
    img = img.astype(np.int32)

    # Take corners of the image
    corners = np.array([points[0,0], points[0,5], points[3,0], points[3,5]])


    black_point_index = np.argmin(img[corners[:,0], corners[:,1]])
    if black_point_index == 0:
        # Rotate the matrix by 180 degrees
        points = np.rot90(points, 2, axes=(0,1))
    elif black_point_index == 1:
        # Rotate by 180 degrees and flip the matrix
        points = np.flip(np.rot90(points, 2, axes=(0,1)), axis=1)
    elif black_point_index == 2:
        # Flip the matrix
        points = np.flip(points, axis=1)
    # else the matrix is already in the correct orientation

    # Now we can associate the correct color to each point
    points = points.reshape(24, 2)
    colors = ["Dark skin", "Light skin", "Blue sky", "Foliage", "Blue flower", "Bluish green", "Orange", "Purplish blue", "Moderate red", "Purple", "Yellow green", "Orange yellow", "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", "White", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black"]
    rgb_triplets = [(115, 82, 68),(194, 150, 130),(98, 122, 157),(87, 108, 67),(133, 128, 177),(103, 189, 170),(214, 126, 44),(80, 91, 166),(193, 90, 99),(94, 60, 108),(157, 188, 64),(224, 163, 46),(56, 61, 150),(70, 148, 73),(175, 54, 60),(231, 199, 31),(187, 86, 149),(8, 133, 161),(243, 243, 242),(200, 200, 200),(160, 160, 160),(122, 122, 122),(85, 85, 85),(52, 52, 52)]
    patches = []
    for i in range(len(points)):
        patch = {
            "color": colors[i],
            "coords": points[i],
            "rgb": rgb_triplets[i]
        }
        patches.append(patch)
    
    return patches



    





    



