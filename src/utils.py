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
from PIL import image
from typing import Dict, List, Tuple, Any


# ==============================
#  Constants
# ==============================
IMAGE_EXTENSION = '.jpg'
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Caption Management Functions
# ============================================

def load_captions_json(json_path):
    """
    Load captions from JSON file
    Args:
        json_path: Path to captions.json file

    Returns:
        Dictionary with image data including captions
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_captions_json(captions, output_path):
    """
    Save generated captions from the VLMs to JSON file.

    Args: 
        captions: Dictionary mapping image filenames to generated captions
        output_path: Path where to save the JSON file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

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
