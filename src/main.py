# main.py
"""
Main script for Adaptive Cellular Automata Edge Detection

This script orchestrates the complete edge detection pipeline:
1. Load images from input directory
2. Apply adaptive CA edge detection
3. Save results and comparisons
4. Generate visual analysis

Usage:
    python main.py --input data/input_images/sample.png --output data/output_images/
    python main.py --input data/input_images/ --output data/output_images/ --batch
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import datetime # Import for timestamp

# Import our modules
from ca import adaptive_ca_edge_detection
from utils import normalize_image, apply_gaussian_filter


def load_image(image_path: str, preprocess: bool = True) -> np.ndarray:
    """
    Load and preprocess an image for edge detection.
    
    Args:
        image_path: Path to input image
        preprocess: Whether to apply preprocessing (Gaussian filter)
        
    Returns:
        Preprocessed grayscale image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to float32 for processing
    image = image.astype(np.float32)
    
    # Optional preprocessing
    if preprocess:
        image = apply_gaussian_filter(image, kernel_size=3, sigma=1.0)
    
    return image


def apply_canny_edge_detection(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    """
    Apply Canny edge detection for comparison.
    
    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny
        
    Returns:
        Binary edge map from Canny detector
    """
    # Convert to uint8 for Canny
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_uint8, low_threshold, high_threshold)
    
    return edges


def create_comparison_plot(
    original: np.ndarray,
    ca_edges: np.ndarray,
    canny_edges: np.ndarray,
    ca_states: np.ndarray,
    iterations: int,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive comparison plot showing results.
    
    Args:
        original: Original grayscale image
        ca_edges: CA edge detection result
        canny_edges: Canny edge detection result
        ca_states: Final CA states for analysis
        iterations: Number of CA iterations
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Adaptive CA Edge Detection Results (Converged in {iterations} iterations)', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # CA Edge Detection
    axes[0, 1].imshow(ca_edges, cmap='gray')
    axes[0, 1].set_title('CA Edge Detection')
    axes[0, 1].axis('off')
    
    # Canny Edge Detection
    axes[0, 2].imshow(canny_edges, cmap='gray')
    axes[0, 2].set_title('Canny Edge Detection')
    axes[0, 2].axis('off')
    
    # CA States visualization
    axes[1, 0].imshow(ca_states, cmap='viridis')
    axes[1, 0].set_title('CA States (0=Non-edge, 1=Potential, 2=Confirmed)')
    axes[1, 0].axis('off')
    
    # Edge comparison overlay
    overlay = np.zeros((*original.shape, 3))
    overlay[:, :, 0] = ca_edges / 255.0  # CA edges in red
    overlay[:, :, 1] = canny_edges / 255.0  # Canny edges in green
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Edge Comparison (Red=CA, Green=Canny)')
    axes[1, 1].axis('off')
    
    # Statistics
    ca_edge_pixels = np.sum(ca_edges > 0)
    canny_edge_pixels = np.sum(canny_edges > 0)
    total_pixels = original.size
    
    stats_text = f"""
    CA Edge Pixels: {ca_edge_pixels} ({ca_edge_pixels/total_pixels*100:.2f}%)
    Canny Edge Pixels: {canny_edge_pixels} ({canny_edge_pixels/total_pixels*100:.2f}%)
    
    CA State Distribution:
    Non-edge (0): {np.sum(ca_states == 0)} ({np.sum(ca_states == 0)/total_pixels*100:.1f}%)
    Potential (1): {np.sum(ca_states == 1)} ({np.sum(ca_states == 1)/total_pixels*100:.1f}%)
    Confirmed (2): {np.sum(ca_states == 2)} ({np.sum(ca_states == 2)/total_pixels*100:.1f}%)
    
    Convergence: {iterations} iterations
    """
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # We explicitly close the plot to prevent it from consuming memory
        # and potentially causing the UserWarning in interactive environments.
        plt.close(fig) 
    
    if save_path is None: # Only show if not saving, or if explicitly requested and not saving
        plt.show()


def process_single_image(
    input_path: str,
    base_output_dir: str, # Renamed to base_output_dir
    T_low: float,
    T_high: float,
    k_std_dev: float,
    max_iterations: int,
    tolerance: float,
    show_plot: bool = True,
    save_plot: bool = True
) -> dict:
    """
    Process a single image through the CA edge detection pipeline.
    
    Args:
        input_path: Path to input image
        base_output_dir: Base directory to save results (a subfolder will be created here)
        T_low: Lower threshold for CA
        T_high: Upper threshold for CA
        k_std_dev: Standard deviation multiplier
        max_iterations: Maximum CA iterations
        tolerance: Convergence tolerance
        show_plot: Whether to display comparison plot
        save_plot: Whether to save comparison plot
        
    Returns:
        Dictionary containing processing results and statistics
    """
    print(f"Processing: {input_path}")
    
    # Create a unique output sub-directory for this image's results
    base_name = Path(input_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct unique output directory: base_output_dir/image_name_timestamp/
    output_dir = Path(base_output_dir) / f"{base_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
    
    print(f"  Saving results to: {output_dir}")

    # Load and preprocess image
    image = load_image(input_path, preprocess=True)
    
    # Apply CA edge detection
    ca_edges, ca_states, iterations = adaptive_ca_edge_detection(
        image, T_low, T_high, k_std_dev, max_iterations, tolerance
    )
    
    # Apply Canny for comparison
    canny_edges = apply_canny_edge_detection(image, T_low, T_high)
    
    # Save results
    ca_output_path = output_dir / f"{base_name}_ca_edges.png"
    canny_output_path = output_dir / f"{base_name}_canny_edges.png"
    states_output_path = output_dir / f"{base_name}_ca_states.png"
    
    cv2.imwrite(str(ca_output_path), ca_edges)
    cv2.imwrite(str(canny_output_path), canny_edges)
    
    # Save CA states as color-coded image
    states_colored = np.zeros((*ca_states.shape, 3), dtype=np.uint8)
    states_colored[ca_states == 0] = [0, 0, 0]      # Black for non-edge
    states_colored[ca_states == 1] = [128, 128, 0]  # Yellow for potential
    states_colored[ca_states == 2] = [255, 255, 255] # White for confirmed
    cv2.imwrite(str(states_output_path), states_colored)
    
    # Create and save comparison plot
    plot_path = output_dir / f"{base_name}_comparison.png" if save_plot else None
    if show_plot or save_plot:
        create_comparison_plot(image, ca_edges, canny_edges, ca_states, iterations, plot_path)
    
    # Calculate statistics
    results = {
        'input_path': input_path,
        'iterations': iterations,
        'ca_edge_pixels': np.sum(ca_edges > 0),
        'canny_edge_pixels': np.sum(canny_edges > 0),
        'total_pixels': image.size,
        'ca_edge_ratio': np.sum(ca_edges > 0) / image.size,
        'canny_edge_ratio': np.sum(canny_edges > 0) / image.size,
        'state_distribution': {
            'non_edge': np.sum(ca_states == 0),
            'potential': np.sum(ca_states == 1),
            'confirmed': np.sum(ca_states == 2)
        },
        'output_folder': str(output_dir) # Add the specific output folder to results
    }
    
    print(f"  Converged in {iterations} iterations")
    print(f"  CA edges: {results['ca_edge_pixels']} pixels ({results['ca_edge_ratio']*100:.2f}%)")
    
    return results


def process_batch(
    input_dir: str,
    base_output_dir: str, # Renamed to base_output_dir
    T_low: float,
    T_high: float,
    k_std_dev: float,
    max_iterations: int,
    tolerance: float
) -> List[dict]:
    """
    Process multiple images in batch mode.
    
    Args:
        input_dir: Directory containing input images
        base_output_dir: Base directory to save results (a single timestamped subfolder will be created here)
        T_low: Lower threshold for CA
        T_high: Upper threshold for CA
        k_std_dev: Standard deviation multiplier
        max_iterations: Maximum CA iterations
        tolerance: Convergence tolerance
        
    Returns:
        List of processing results for each image
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create a unique output sub-directory for this batch run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = Path(base_output_dir) / f"batch_run_{timestamp}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Batch run output will be saved to: {batch_output_dir}")

    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in: {input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
        
        try:
            # Pass the *specific* batch output directory here
            result = process_single_image(
                str(image_file), str(batch_output_dir), T_low, T_high, k_std_dev, # Pass batch_output_dir
                max_iterations, tolerance, show_plot=False, save_plot=True
            )
            results.append(result)
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            continue
    
    # Print batch summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    if results:
        avg_iterations = np.mean([r['iterations'] for r in results])
        avg_ca_ratio = np.mean([r['ca_edge_ratio'] for r in results])
        avg_canny_ratio = np.mean([r['canny_edge_ratio'] for r in results])
        
        print(f"Successfully processed: {len(results)}/{len(image_files)} images")
        print(f"Average iterations: {avg_iterations:.1f}")
        print(f"Average CA edge ratio: {avg_ca_ratio*100:.2f}%")
        print(f"Average Canny edge ratio: {avg_canny_ratio*100:.2f}%")
        print(f"All batch results saved under: {batch_output_dir}")
    else:
        print("No images were successfully processed.")
    
    return results


def main():
    """Main function with argument parsing and orchestration."""
    parser = argparse.ArgumentParser(
        description="Adaptive Cellular Automata Edge Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single image (creates 'data/output_images/image_name_timestamp/' for output):
    python main.py --input data/input_images/sample.png --output data/output_images/
  
  Process batch of images (creates 'data/output_images/batch_run_timestamp/' for output):
    python main.py --input data/input_images/ --output data/output_images/ --batch
  
  Custom parameters:
    python main.py --input sample.png --output results/ --T_low 15 --T_high 30 --iterations 100
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', '-o', required=True,
                       help='Base output directory for results') # Clarified help text
    
    # Processing mode
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all images in input directory')
    
    # CA parameters
    parser.add_argument('--T_low', type=float, default=10,
                       help='Lower threshold for edge detection (default: 10)')
    parser.add_argument('--T_high', type=float, default=25,
                       help='Upper threshold for edge detection (default: 25)')
    parser.add_argument('--k_std_dev', type=float, default=1.0,
                       help='Standard deviation multiplier (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Maximum CA iterations (default: 50)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Convergence tolerance (default: 0.01)')
    
    # Display options
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable interactive plots')
    parser.add_argument('--no-save-plot', action='store_true',
                       help='Disable saving comparison plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.T_low >= args.T_high:
        print("Error: T_low must be less than T_high")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
    
    # Create the base output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            results = process_batch(
                args.input, args.output, args.T_low, args.T_high,
                args.k_std_dev, args.iterations, args.tolerance
            )
        else:
            # Single image processing
            result = process_single_image(
                args.input, args.output, args.T_low, args.T_high, # Pass base_output_dir
                args.k_std_dev, args.iterations, args.tolerance,
                show_plot=not args.no_plot,
                save_plot=not args.no_save_plot
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()