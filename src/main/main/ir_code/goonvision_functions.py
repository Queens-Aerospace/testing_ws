import numpy as np
from scipy.optimize import minimize
#tau is range in meters
def IRemmiterExact(super_cluster, estimated_centers, tau):
    N = estimated_centers
    emmiter_locations = np.zeros((N, 3))
    
    for i in range(N):
        best_cluster = []
        temp_center = np.copy(estimated_centers[i, :])
        
        # Vary the center point z-component from 1m through 30m
        for l in range(1, 31):
            cluster = []
            temp_center[2] = l  # Set the z-component of the center point
            
            dist = distance_to_all_lines(temp_center, super_cluster)
            
            for j in range(super_cluster.shape[0]):
                if dist[j] <= tau:
                    cluster.append(super_cluster[j, :])
            
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        
        # Add exact centers to the output matrix
        center, _ = dist_squared_minimizer(best_cluster)
        emmiter_locations[i, :3] = center
    
    return emmiter_locations

def distance_point_to_line(pt, p_i, d_i):
    """
    Calculates the distance from a point `pt` to a line defined by the point `p_i` and direction `d_i`.
    """
    # Vector from p_i to pt
    v = pt - p_i
    # Cross product magnitude / length(d_i) if d_i is not unit
    # Hopefully, d_i is unit, but let's be safe
    cross_prod = np.cross(v, d_i)
    d = np.linalg.norm(cross_prod) / np.linalg.norm(d_i)
    return d

def distance_to_all_lines(pt, lines):
    """
    Computes the distances from point `pt` to all lines in `lines`.
    lines: S x 6 matrix where each row contains the start point (3 elements) and direction vector (3 elements).
    pt: A 1x3 point.
    """
    S = lines.shape[0]
    dists = np.zeros(S)
    
    for i in range(S):
        p_i = lines[i, :3]  # Start point of the line
        d_i = lines[i, 3:]  # Direction vector of the line
        dists[i] = distance_point_to_line(pt, p_i, d_i)
    
    return dists

def dist_squared_minimizer(lines):
    """
    Finds the point that is closest to all given lines.
    """
    N = lines.shape[0]
    print('Lines:')
    print(lines)
    
    # Normalize the direction vectors
    Do = normalize_vectors(lines[:, 3:6])
    Po = lines[:, :3]
    
    print('Do:')
    print(Do)

    # Starting guess is the mean of Po
    x0 = np.mean(Po, axis=0)
    
    # Define the objective function (sum of squared distances)
    objective = lambda x: sum_of_squared_distances(x, Po, Do)
    
    # Use scipy's minimize function to find the best point
    result = minimize(objective, x0, options={'disp': False})
    x_best = result.x
    
    residuals = distance_point_to_lines(x_best, Po, Do)  # Computes distances from x_best to each line
    
    return x_best, residuals

def normalize_vectors(Do):
    """
    Normalizes the rows of the matrix to unit vectors.
    """
    magnitudes = np.sqrt(np.sum(Do**2, axis=1))  # Compute the magnitude of each row
    if np.any(magnitudes == 0):
        raise ValueError("One or more direction vectors have zero magnitude and cannot be normalized.")
    D_normalized = Do / magnitudes[:, None]  # Normalize each row
    return D_normalized

def sum_of_squared_distances(x, Po, Do):
    """
    Squared distances - critical for optimization algorithm.
    """
    dist_vals = distance_point_to_lines(x, Po, Do)
    return np.sum(dist_vals**2)

def distance_point_to_lines(x, Po, Do):
    """
    Determines the distances from the point `x` to all lines.
    """
    N = Po.shape[0]
    x_mat = np.tile(x, (N, 1))  # Replicate `x` to match size of Po
    cross_prod = np.cross(x_mat - Po, Do)  # Cross product for each line
    dist_vals = np.sqrt(np.sum(cross_prod**2, axis=1))  # Distance is the magnitude of the cross product
    return dist_vals

def emitters_to_coordinates(M):
    """
    Takes in a 1100x1100 matrix M with a small number (2-8) of 1's 
    (representing IR emitters). Returns an N x 3 matrix 'coords', 
    where each row is [x, y, 1]. (0,0) is taken as the center of the matrix/circle.
    """
    # --- 1) Find the indices of all the 1's ---
    rows, cols = np.where(M == 1)
    
    # The number of emitters
    N = len(rows)
    
    # --- 2) Determine how to transform matrix indices to (x, y) ---
    # The matrix is 1100 x 1100 and represents a 100m x 100m region.
    # => each cell ~ 100/1100 = 0.090909... meters in size.
    
    scale = 100 / 1100  # scale from matrix index distance to meters
    
    # The center of the matrix
    row_center = (1100 + 1) / 2
    col_center = (1100 + 1) / 2
    
    # --- 3) Convert each (row, col) to an (x, y) coordinate ---
    x = (cols - col_center) * scale
    y = (rows - row_center) * scale
    
    # --- 4) Form the Nx3 output matrix. The third column is '1' for convenience ---
    coords = np.column_stack((x, y, np.ones(N)))
    
    return coords

