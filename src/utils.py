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

def save_captions_json(captions, output_path):
    """
    Save generated captions from the VLMs to JSON file.

    Args:
        captions: Dictionary mapping image filenames to generated captions
        output_path: Path where to save the JSON file
    """
    ensure_output_dir(Path(output_path).parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)


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
    image_paths = dataset_path.rglob(f'*.{IMAGE_EXTENSION}')

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


def clear_gpu_cache():
    """
    Clear GPU memory cache to free up VRAM.

    Useful when switching between models or after inference.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
