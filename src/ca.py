# ca.py
"""
Adaptive Cellular Automata Edge Detection

This module implements edge detection using a 3-state cellular automaton that
adapts to local image characteristics. The CA states are:
- State 0: Non-edge (background)
- State 1: Potential edge (candidate)
- State 2: Confirmed edge

Author: Edge Detection CA Project
"""

import numpy as np
from typing import Tuple
from numba import jit # Import numba's jit decorator

from utils import (
    get_moore_neighbors_intensities,
    get_moore_neighbors_states,
    calculate_standard_deviation,
    calculate_adaptive_gaussian_threshold,
    count_state
)

# Apply JIT compilation to these computationally intensive functions
@jit(nopython=True, cache=True)
def initialize_ca_states(
    image: np.ndarray, 
    T_low: float, 
    T_high: float, 
    k_std_dev: float = 1.0
) -> np.ndarray:
    """
    Initialize the cellular automaton states based on local image characteristics.
    
    For each pixel, calculates adaptive threshold using Gaussian-weighted neighbors
    and assigns initial state based on gradient magnitude.
    
    Args:
        image: 2D grayscale image array
        T_low: Lower threshold for edge detection
        T_high: Upper threshold for edge detection
        k_std_dev: Standard deviation multiplier for adaptive thresholding
        
    Returns:
        2D array of initial CA states (0, 1, or 2)
    """
    h, w = image.shape
    states = np.zeros((h, w), dtype=np.uint8)
    
    # Numba requires these helper functions to be JIT compiled or rewritten within the JIT function
    # For now, we'll assume utils functions are suitable or will be inlined/optimized by Numba.
    # For optimal Numba performance, these nested calls should also be JIT-compiled or optimized.
    # Note: Numba's nopython mode has limitations on what Python features it supports.
    # List appending and np.array conversion might be slow, but it's a start.

    # To make Numba work best here, we should inline the neighbor logic
    # or ensure `utils` functions are also JIT compiled and work in nopython mode.
    # For simplicity, let's keep the existing structure and rely on Numba's
    # ability to optimize Python loops.

    # --- Inlining neighborhood logic for better Numba performance ---
    # This avoids list creation and function calls inside the hot loop.
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1], [1, 0], [1, 1]
    ], dtype=np.int8)

    # Pre-calculate Gaussian weights for adaptive threshold
    gaussian_weights_flat = np.array([
        0.0751, 0.1238, 0.0751,
        0.1238,         0.1238,
        0.0751, 0.1238, 0.0751
    ], dtype=np.float32)
    gaussian_weights_flat /= np.sum(gaussian_weights_flat) # Normalize

    for i in range(h):
        for j in range(w):
            current_intensity = image[i, j]
            
            # Manually collect neighbors (Numba-friendly way)
            neighbors_int_list = np.empty(8, dtype=np.float32)
            idx = 0
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                # Handle boundary conditions with edge padding
                ni = max(0, min(h - 1, ni))
                nj = max(0, min(w - 1, nj))
                neighbors_int_list[idx] = image[ni, nj]
                idx += 1

            # Numba-friendly standard deviation and adaptive threshold
            local_std = np.std(neighbors_int_list) if len(neighbors_int_list) > 0 else 0.0
            
            weighted_mean = np.sum(neighbors_int_list * gaussian_weights_flat)
            threshold = weighted_mean + (k_std_dev * local_std)
            
            gradient = np.abs(current_intensity - threshold)
            
            if gradient > T_high:
                states[i, j] = 2  # Confirmed edge
            elif gradient > T_low:
                states[i, j] = 1  # Potential edge
            else:
                states[i, j] = 0  # Non-edge
                
    return states


@jit(nopython=True, cache=True)
def apply_ca_rules(
    image: np.ndarray, 
    states: np.ndarray, 
    T_low: float, 
    T_high: float, 
    k_std_dev: float = 1.0
) -> np.ndarray:
    """
    Apply cellular automaton transition rules for one iteration.
    
    Each cell's next state depends on:
    - Local gradient magnitude (adaptive threshold)
    - Current state
    - Neighboring edge states (connectivity)
    
    Args:
        image: 2D grayscale image array
        states: Current CA states
        T_low: Lower threshold for edge detection
        T_high: Upper threshold for edge detection
        k_std_dev: Standard deviation multiplier for adaptive thresholding
        
    Returns:
        2D array of new CA states after applying transition rules
    """
    h, w = states.shape
    new_states = states.copy()  # Simultaneous update
    
    # --- Inlining neighborhood logic for better Numba performance ---
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1], [1, 0], [1, 1]
    ], dtype=np.int8)

    # Pre-calculate Gaussian weights for adaptive threshold
    gaussian_weights_flat = np.array([
        0.0751, 0.1238, 0.0751,
        0.1238,         0.1238,
        0.0751, 0.1238, 0.0751
    ], dtype=np.float32)
    gaussian_weights_flat /= np.sum(gaussian_weights_flat) # Normalize

    for i in range(h):
        for j in range(w):
            current_state = states[i, j]
            current_intensity = image[i, j]

            # Manually collect neighbors (Numba-friendly way)
            neighbors_int_list = np.empty(8, dtype=np.float32)
            neighbors_states_list = np.empty(8, dtype=np.uint8)
            idx = 0
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                # Handle boundary conditions with edge padding
                ni = max(0, min(h - 1, ni))
                nj = max(0, min(w - 1, nj))
                neighbors_int_list[idx] = image[ni, nj]
                neighbors_states_list[idx] = states[ni, nj]
                idx += 1
            
            # Numba-friendly standard deviation and adaptive threshold
            local_std = np.std(neighbors_int_list) if len(neighbors_int_list) > 0 else 0.0
            weighted_mean = np.sum(neighbors_int_list * gaussian_weights_flat)
            threshold = weighted_mean + (k_std_dev * local_std)
            gradient = np.abs(current_intensity - threshold)
            
            # Count neighboring edge states for connectivity (Numba-friendly count)
            edge_count = 0
            potential_count = 0
            for state in neighbors_states_list:
                if state == 2:
                    edge_count += 1
                elif state == 1:
                    potential_count += 1
            
            # Apply state transition rules
            if current_state == 0:  # Non-edge
                if gradient > T_high or (gradient > T_low and edge_count >= 2):
                    new_states[i, j] = 1 # Promote to potential
                    
            elif current_state == 1:  # Potential edge
                if gradient > T_high and edge_count >= 1:
                    new_states[i, j] = 2 # Promote to confirmed edge
                elif gradient < T_low and edge_count == 0:
                    new_states[i, j] = 0 # Demote to non-edge if weak gradient and isolated
                    
            elif current_state == 2:  # Confirmed edge
                if gradient < T_low and edge_count <= 1:
                    new_states[i, j] = 1 # Demote to potential if weakening
                elif gradient < T_low * 0.5 and edge_count == 0:
                    new_states[i, j] = 0 # Demote to non-edge if very weak and isolated
                    
    return new_states


@jit(nopython=True, cache=True) # JIT this as well
def has_converged(
    old_states: np.ndarray, 
    new_states: np.ndarray, 
    tolerance: float = 0.01
) -> bool:
    """
    Check if the cellular automaton has converged.
    
    Convergence is determined by the ratio of changed cells falling
    below the specified tolerance threshold.
    
    Args:
        old_states: Previous iteration states
        new_states: Current iteration states
        tolerance: Convergence threshold (ratio of changed cells)
        
    Returns:
        True if system has converged, False otherwise
    """
    total_cells = old_states.size
    changed_cells = np.sum(old_states != new_states)
    change_ratio = changed_cells / total_cells
    
    return change_ratio < tolerance


def adaptive_ca_edge_detection(
    image: np.ndarray,
    T_low: float = 10,
    T_high: float = 25,
    k_std_dev: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform adaptive cellular automata edge detection on a grayscale image.
    
    This is the main function that orchestrates the entire edge detection process:
    1. Initialize CA states based on local image characteristics
    2. Iteratively apply CA transition rules until convergence
    3. Extract final edge map from confirmed edge states
    
    Args:
        image: Input grayscale image (2D numpy array)
        T_low: Lower threshold for edge detection
        T_high: Upper threshold for edge detection  
        k_std_dev: Standard deviation multiplier for adaptive thresholding
        max_iterations: Maximum number of CA iterations
        tolerance: Convergence threshold (ratio of changed cells)
        
    Returns:
        Tuple containing:
        - edge_map: Binary edge map (0 or 255)
        - final_states: Final CA states for analysis
        - iterations: Number of iterations until convergence
        
    Raises:
        ValueError: If input image is not 2D grayscale
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image (2D array)")
    
    image = image.astype(np.float32)
    
    states = initialize_ca_states(image, T_low, T_high, k_std_dev)
    
    for iteration in range(max_iterations):
        new_states = apply_ca_rules(image, states, T_low, T_high, k_std_dev)
        
        if has_converged(states, new_states, tolerance):
            break
            
        states = new_states
    
    edge_map = (states == 2).astype(np.uint8) * 255
    
    return edge_map, states, iteration + 1