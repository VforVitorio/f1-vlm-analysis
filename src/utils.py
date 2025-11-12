"""
Utility functions for VLM caption generation and evaluation

This module provides helper functions for:
- Loading and saving captions in JSON format
- Image loading and path management
- Metrics storage and retrieval
- Progress display and formatted output
- GPU device management and memory cleanup

"""

import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any


# ==============================
#  Constants
# ==============================
IMAGE_EXTENSION = '.jpg'
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Caption Management Functions
# ============================================

def get_incremental_filename(base_dir, base_name, extension=".json"):
    """
    Generate incremental filename to avoid overwriting existing files.

    Creates filenames in the format:
    - base_name.extension (if doesn't exist)
    - base_name_1.extension
    - base_name_2.extension
    - base_name_N.extension

    Args:
        base_dir: Directory where file will be saved (Path or str)
        base_name: Base filename without extension (e.g., "generated_captions")
        extension: File extension including dot (default: ".json")

    Returns:
        Path: Full path with incremental number if needed

    Example:
        >>> get_incremental_filename("results/blip", "generated_captions")
        Path("results/blip/generated_captions.json")  # if doesn't exist
        Path("results/blip/generated_captions_1.json")  # if base exists
        Path("results/blip/generated_captions_2.json")  # if _1 exists
    """
    base_dir = Path(base_dir)

    # Try base filename first
    output_path = base_dir / f"{base_name}{extension}"
    if not output_path.exists():
        return output_path

    # Find next available incremental number
    iteration = 1
    while True:
        output_path = base_dir / f"{base_name}_{iteration}{extension}"
        if not output_path.exists():
            return output_path
        iteration += 1


def save_captions_json(captions, output_path, incremental=True):
    """
    Save generated captions from the VLMs to JSON file.

    Args:
        captions: Dictionary mapping image filenames to generated captions
        output_path: Path where to save the JSON file (str or Path)
        incremental: If True, creates incremental filenames to avoid overwriting.
                    If False, overwrites existing file. Default: True

    Returns:
        Path: Actual path where file was saved (may differ if incremental=True)
    """
    output_path = Path(output_path)

    if incremental:
        # Generate incremental filename
        base_name = output_path.stem  # filename without extension
        extension = output_path.suffix  # .json
        base_dir = output_path.parent

        output_path = get_incremental_filename(base_dir, base_name, extension)

    # Ensure directory exists
    ensure_output_dir(output_path.parent)

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"✅ Captions saved to: {output_path}")
    return output_path

    return output_path


def format_captions_for_evaluation(ground_truth_data, generated_captions):
    """
    Format captions for evaluation metrics.

    Converts captions to the format expected by evaluation metrics:
    - references: list of lists (each image can have multiple reference captions)
    - predictions: list of strings (one generated caption per image)

    Args:
        ground_truth_data: Dictionary from captions.json with ground truth
        generated_captions: Dictionary mapping filenames to generated captions

    Returns:
        Tuple of (references, predictions) formatted for evaluation
    """
    references = []
    predictions = []

    for image_info in ground_truth_data['images']:
        filename = image_info['filename']

        # Ground truth caption (wrapped in list for multiple references support)
        references.append([image_info['caption']])

        # Generated caption
        if filename in generated_captions:
            predictions.append(generated_captions[filename])
        else:
            predictions.append("")  # Empty if not generated

    return references, predictions


# ============================================
# Image Management Functions
# ============================================

def load_image(image_path):
    """
    Load image from file as PIL Image.

    Args:
        image_path: Path to the image file

    Returns:
        PIL IMage object in RGB mode

    """

    image = Image.open(image_path).convert("RGB")

    return image


def get_image_paths(dataset_dir):
    """
    Get all image file paths from dataset directory.

    Searches recursively through all subdirectories for .jpg files.

    Args:
        dataset_dir: Path to dataset root directory

    Returns:
        Sorted list of image file paths as strings
    """
    dataset_path = Path(dataset_dir)

    # Find all .jpg files recursively
    image_paths = dataset_path.rglob(f'*{IMAGE_EXTENSION}')

    # Convert to strings and sort
    image_paths_list = [str(path) for path in image_paths]
    image_paths_sorted = sorted(image_paths_list)

    return image_paths_sorted

# ============================================
# Results Management Functions
# ============================================


def save_metrics(metrics, output_path):
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary containing metrics names and values
        output_path: Path where to save the metrics JSON file
    """
    ensure_output_dir(Path(output_path).parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(metrics_path):
    """
    Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Dictionary with metric names and values
    """

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def ensure_output_dir(output_dir):
    """
    Create output directory if it does not exist.

    Args:
        output_dir: Path to directory to create
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


# ============================================
# Display Utilities
# ============================================

def print_progress(current, total, image_name):
    """
    Display progress of image processing.


    Args:
        current: Current image number being processed
        total: Total number of images
        image_name: Name of current image file
    """

    print(f"Processing image {current}/{total}: {image_name}")


def print_metrics_summary(metrics):
    """
    Print evaluation metrics in formatted table.

    Args:
        metrics: Dictionary containing metric names and values
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:20s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:20s}: {metric_value}")

    print("="*60 + "\n")


# ============================================
# Device Utilities
# ============================================

def get_device():
    """
    Detect and return available device for inference.

    Returns:
        torch.device object (cuda or cpu)
    """
    device = torch.device(DEFAULT_DEVICE)

    if DEFAULT_DEVICE == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    return device


def get_device_from_arg(device):
    """
    Convert device argument to torch.device object.

    If device is None, auto-detect CUDA availability.
    Used by model initialization to standardize device handling.

    Args:
        device: torch device ('cuda', 'cpu', or None for auto-detect)

    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def clear_gpu_cache():
    """
    Clear GPU memory cache to free up VRAM.

    Useful when switching between models or after inference.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
