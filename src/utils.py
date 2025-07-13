"""
Utility functions for adaptive cellular automata edge detection.

This module provides helper functions for neighborhood operations,
statistical calculations, and adaptive thresholding.
"""

import numpy as np
from typing import List, Tuple


def get_moore_neighbors_intensities(image: np.ndarray, i: int, j: int) -> List[float]:
    """
    Get intensity values of Moore neighborhood (8-connected) around pixel (i,j).
    
    Handles boundary conditions by padding with edge values.
    
    Args:
        image: 2D grayscale image array
        i: Row index of center pixel
        j: Column index of center pixel
        
    Returns:
        List of intensity values from 8-connected neighborhood
    """
    h, w = image.shape
    neighbors = []
    
    # Define 8-connected neighborhood offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
               (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for di, dj in offsets:
        ni, nj = i + di, j + dj
        
        # Handle boundary conditions with edge padding
        ni = max(0, min(h - 1, ni))
        nj = max(0, min(w - 1, nj))
        
        neighbors.append(float(image[ni, nj]))
    
    return neighbors


def get_moore_neighbors_states(states: np.ndarray, i: int, j: int) -> List[int]:
    """
    Get CA state values of Moore neighborhood (8-connected) around pixel (i,j).
    
    Handles boundary conditions by padding with edge values.
    
    Args:
        states: 2D array of CA states
        i: Row index of center pixel
        j: Column index of center pixel
        
    Returns:
        List of state values from 8-connected neighborhood
    """
    h, w = states.shape
    neighbors = []
    
    # Define 8-connected neighborhood offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
               (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for di, dj in offsets:
        ni, nj = i + di, j + dj
        
        # Handle boundary conditions with edge padding
        ni = max(0, min(h - 1, ni))
        nj = max(0, min(w - 1, nj))
        
        neighbors.append(int(states[ni, nj]))
    
    return neighbors


def calculate_standard_deviation(neighbors: List[float]) -> float:
    """
    Calculate standard deviation of neighborhood intensity values.
    
    Args:
        neighbors: List of intensity values from neighborhood
        
    Returns:
        Standard deviation of the neighborhood intensities
    """
    if len(neighbors) == 0:
        return 0.0
        
    neighbors_array = np.array(neighbors)
    return float(np.std(neighbors_array))


def calculate_adaptive_gaussian_threshold(neighbors: List[float], k_std_dev: float = 1.0) -> float:
    """
    Calculate adaptive threshold using Gaussian-weighted mean of neighborhood.
    
    Uses a 3x3 Gaussian kernel to weight the neighborhood values, with the
    center pixel having the highest weight.
    
    Args:
        neighbors: List of 8 neighborhood intensity values (Moore neighborhood)
        k_std_dev: Standard deviation multiplier for threshold adaptation
        
    Returns:
        Gaussian-weighted threshold value
    """
    if len(neighbors) != 8:
        # Fallback to simple mean if neighborhood size is unexpected
        return float(np.mean(neighbors))
    
    # 3x3 Gaussian kernel (center weight is implicit for current pixel)
    # Arranged as: [TL, T, TR, L, R, BL, B, BR]
    gaussian_weights = np.array([
        0.0751, 0.1238, 0.0751,  # Top row
        0.1238,         0.1238,  # Middle row (center excluded)
        0.0751, 0.1238, 0.0751   # Bottom row
    ])
    
    # Normalize weights to sum to 1
    gaussian_weights = gaussian_weights / np.sum(gaussian_weights)
    
    # Calculate weighted mean
    neighbors_array = np.array(neighbors)
    weighted_mean = np.sum(neighbors_array * gaussian_weights)
    
    # Add adaptive component based on local standard deviation
    local_std = calculate_standard_deviation(neighbors)
    adaptive_threshold = weighted_mean + (k_std_dev * local_std)
    
    return float(adaptive_threshold)


def count_state(neighbors_states: List[int], target_state: int) -> int:
    """
    Count occurrences of a specific state in the neighborhood.
    
    Args:
        neighbors_states: List of CA state values from neighborhood
        target_state: State value to count (0, 1, or 2)
        
    Returns:
        Number of neighbors with the target state
    """
    return sum(1 for state in neighbors_states if state == target_state)


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian filter to image for noise reduction (optional utility).
    
    Args:
        image: Input 2D grayscale image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma)


def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image: Input image array
        target_range: Target min and max values (min, max)
        
    Returns:
        Normalized image
    """
    min_val, max_val = target_range
    img_min, img_max = image.min(), image.max()
    
    if img_max == img_min:
        return np.full_like(image, min_val)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized