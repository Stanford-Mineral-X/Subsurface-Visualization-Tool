import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf

# Gaussian Process Smoothing (Basic Implementation)


def gaussian_smoothing(x, y, z, length_scale=1.0):
    """
    Apply Gaussian smoothing to Z values using a Radial Basis Function (RBF) interpolator.
    """
    rbf = Rbf(x, y, z, function='gaussian', epsilon=length_scale)
    return rbf(x, y)

# Fast Nearest Neighbor Interpolation


def nearest_neighbor_interpolation(x, y, z, grid_x, grid_y):
    """
    Interpolates Z values using nearest neighbor approach.
    """
    tree = cKDTree(np.column_stack((x, y)))
    _, indices = tree.query(np.column_stack((grid_x.ravel(), grid_y.ravel())))
    return z[indices].reshape(grid_x.shape)

# Custom Contour Adjustment


def adjust_contours(Z, threshold):
    """
    Apply a threshold to contour data to highlight specific ranges.
    """
    return np.where(Z > threshold, Z, np.nan)

# Data Normalization


def normalize_data(z):
    """
    Normalize data to range [0,1] for consistency in visualization.
    """
    return (z - np.min(z)) / (np.max(z) - np.min(z))


if __name__ == "__main__":
    print("Utility functions for subsurface data processing.")
