"""
Unit tests for adaptive cellular automata edge detection.

Tests cover core functionality, boundary conditions, and edge cases.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ca import (
    initialize_ca_states,
    apply_ca_rules,
    has_converged,
    adaptive_ca_edge_detection
)
from utils import (
    get_moore_neighbors_intensities,
    get_moore_neighbors_states,
    calculate_standard_deviation,
    calculate_adaptive_gaussian_threshold,
    count_state
)


class TestUtils:
    """Test utility functions."""
    
    def test_get_moore_neighbors_intensities_center(self):
        """Test neighborhood extraction for center pixel."""
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        
        neighbors = get_moore_neighbors_intensities(image, 1, 1)
        expected = [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]
        assert neighbors == expected
    
    def test_get_moore_neighbors_intensities_corner(self):
        """Test neighborhood extraction for corner pixel (boundary handling)."""
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        
        neighbors = get_moore_neighbors_intensities(image, 0, 0)
        # Corner pixel should have padded neighbors
        assert len(neighbors) == 8
        assert neighbors[0] == 1.0  # Top-left (padded)
        assert neighbors[1] == 1.0  # Top (padded)
        assert neighbors[4] == 2.0  # Right
    
    def test_get_moore_neighbors_states(self):
        """Test CA state neighborhood extraction."""
        states = np.array([
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ], dtype=np.uint8)
        
        neighbors = get_moore_neighbors_states(states, 1, 1)
        expected = [0, 1, 2, 1, 0, 2, 0, 1]
        assert neighbors == expected
    
    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        neighbors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        std_dev = calculate_standard_deviation(neighbors)
        expected_std = np.std(neighbors)
        assert abs(std_dev - expected_std) < 1e-6
    
    def test_calculate_standard_deviation_empty(self):
        """Test standard deviation with empty input."""
        std_dev = calculate_standard_deviation([])
        assert std_dev == 0.0
    
    def test_calculate_adaptive_gaussian_threshold(self):
        """Test adaptive Gaussian threshold calculation."""
        neighbors = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        threshold = calculate_adaptive_gaussian_threshold(neighbors, k_std_dev=1.0)
        # Should be weighted mean plus std deviation component
        assert isinstance(threshold, float)
        assert threshold > 0
    
    def test_count_state(self):
        """Test state counting in neighborhood."""
        neighbors_states = [0, 1, 2, 0, 1, 2, 0, 1]
        
        count_0 = count_state(neighbors_states, 0)
        count_1 = count_state(neighbors_states, 1)
        count_2 = count_state(neighbors_states, 2)
        
        assert count_0 == 3
        assert count_1 == 3
        assert count_2 == 2


class TestCACore:
    """Test core CA functions."""
    
    def test_initialize_ca_states_basic(self):
        """Test CA state initialization."""
        image = np.array([
            [0, 100, 0],
            [100, 200, 100],
            [0, 100, 0]
        ], dtype=np.float32)
        
        states = initialize_ca_states(image, T_low=10, T_high=50)
        
        # Check that states are in valid range
        assert np.all(states >= 0)
        assert np.all(states <= 2)
        assert states.shape == image.shape
    
    def test_initialize_ca_states_uniform(self):
        """Test CA initialization with uniform image."""
        image = np.full((5, 5), 128.0, dtype=np.float32)
        states = initialize_ca_states(image, T_low=10, T_high=50)
        
        # Uniform image should result in mostly non-edge states
        assert np.all(states == 0)
    
    def test_apply_ca_rules_basic(self):
        """Test CA rule application."""
        image = np.array([
            [0, 100, 0],
            [100, 200, 100],
            [0, 100, 0]
        ], dtype=np.float32)
        
        states = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0]
        ], dtype=np.uint8)
        
        new_states = apply_ca_rules(image, states, T_low=10, T_high=50)
        
        # Check that output is valid
        assert new_states.shape == states.shape
        assert np.all(new_states >= 0)
        assert np.all(new_states <= 2)
    
    def test_has_converged_identical(self):
        """Test convergence detection with identical states."""
        states = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
        assert has_converged(states, states, tolerance=0.01) == True
    
    def test_has_converged_different(self):
        """Test convergence detection with different states."""
        old_states = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
        new_states = np.array([[1, 1, 2], [1, 2, 1]], dtype=np.uint8)
        
        # 2 out of 6 cells changed = 33% > 1% tolerance
        assert has_converged(old_states, new_states, tolerance=0.01) == False
        
        # But should converge with higher tolerance
        assert has_converged(old_states, new_states, tolerance=0.5) == True


class TestEdgeDetection:
    """Test complete edge detection pipeline."""
    
    def test_adaptive_ca_edge_detection_basic(self):
        """Test complete edge detection on simple image."""
        # Create simple edge image
        image = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 255, 255, 255],
            [0, 0, 255, 255, 255],
            [0, 0, 255, 255, 255]
        ], dtype=np.float32)
        
        edge_map, final_states, iterations = adaptive_ca_edge_detection(
            image, T_low=10, T_high=50, max_iterations=10
        )
        
        # Check output validity
        assert edge_map.shape == image.shape
        assert final_states.shape == image.shape
        assert isinstance(iterations, int)
        assert iterations >= 1
        
        # Edge map should be binary
        assert np.all((edge_map == 0) | (edge_map == 255))
    
    def test_adaptive_ca_edge_detection_uniform(self):
        """Test edge detection on uniform image (should find no edges)."""
        image = np.full((10, 10), 128.0, dtype=np.float32)
        
        edge_map, final_states, iterations = adaptive_ca_edge_detection(
            image, T_low=10, T_high=50, max_iterations=10
        )
        
        # Uniform image should have no edges
        assert np.all(edge_map == 0)
        assert np.all(final_states == 0)
    
    def test_adaptive_ca_edge_detection_single_line(self):
        """Test detection of single horizontal line."""
        image = np.zeros((5, 5), dtype=np.float32)
        image[2, :] = 255  # Horizontal line in middle
        
        edge_map, final_states, iterations = adaptive_ca_edge_detection(
            image, T_low=20, T_high=100, max_iterations=20
        )
        
        # Should detect some edges along the line
        assert np.sum(edge_map) > 0
        assert iterations <= 20
    
    def test_adaptive_ca_edge_detection_invalid_input(self):
        """Test error handling for invalid input."""
        # Test 3D image (should raise error)
        image_3d = np.random.rand(10, 10, 3)
        
        with pytest.raises(ValueError, match="Input must be grayscale image"):
            adaptive_ca_edge_detection(image_3d)
    
    def test_adaptive_ca_edge_detection_parameters(self):
        """Test different parameter combinations."""
        image = np.array([
            [0, 0, 255, 0, 0],
            [0, 0, 255, 0, 0],
            [255, 255, 255, 255, 255],
            [0, 0, 255, 0, 0],
            [0, 0, 255, 0, 0]
        ], dtype=np.float32)
        
        # Test with different threshold values
        edge_map_low, _, _ = adaptive_ca_edge_detection(
            image, T_low=5, T_high=20, max_iterations=10
        )
        
        edge_map_high, _, _ = adaptive_ca_edge_detection(
            image, T_low=50, T_high=100, max_iterations=10
        )
        
        # Lower thresholds should generally detect more edges
        assert np.sum(edge_map_low) >= np.sum(edge_map_high)


class TestBoundaryConditions:
    """Test boundary condition handling."""
    
    def test_small_image_1x1(self):
        """Test with 1x1 image."""
        image = np.array([[128]], dtype=np.float32)
        
        edge_map, final_states, iterations = adaptive_ca_edge_detection(
            image, T_low=10, T_high=50, max_iterations=5
        )
        
        assert edge_map.shape == (1, 1)
        assert final_states.shape == (1, 1)
        assert iterations >= 1
    
    def test_small_image_2x2(self):
        """Test with 2x2 image."""
        image = np.array([
            [0, 255],
            [255, 0]
        ], dtype=np.float32)
        
        edge_map, final_states, iterations = adaptive_ca_edge_detection(
            image, T_low=10, T_high=50, max_iterations=5
        )
        
        assert edge_map.shape == (2, 2)
        assert final_states.shape == (2, 2)
        assert iterations >= 1


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])