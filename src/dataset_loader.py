"""
F1 Dataset loader for Practice 3

This module handles loading the F1 image dataset organized in 4 thematic categories.
Provides functions to load images, captions, and organize data by category.

Dataset structure:
    dataset/
    ├── 1_drivers_emotions/
    ├── 2_pit_stops/
    ├── 3_cars_tracks_moments/
    ├── 4_strategy_data/
    └── captions.json

The captions.json file contains ground truth captions for all images,
with fields: id, category, filename, caption.
"""

import json
from pathlib import Path
from typing import Dict, List


# ============================================
# Constants
# ============================================
DATASET_ROOT = "dataset"
CAPTIONS_FILE = "captions.json"
IMAGE_EXTENSION = ".jpg"

CATEGORIES = [
    "1_drivers_emotions",
    "2_pit_stops",
    "3_cars_tracks_moments",
    "4_strategy_data"
]

# ============================================
# Functions
# ============================================


def load_captions(dataset_path):
    """
    Load captions.json file from dataset.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        dict: Captions data with 'images' list containing image metadata
    """
    captions_path = Path(dataset_path) / CAPTIONS_FILE

    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_dataset(dataset_path):
    """
    Load complete F1 dataset with images and captions.


    Args:
        dataset_path: Path to dataset root directory

    Returns:
        dict: Dataset organized by categories with image paths and captions
    """

    dataset_path = Path(dataset_path)

    # Load captions
    captions_data = load_captions(dataset_path)

    # Organize by categories
    dataset = {
        "images": captions_data["images"],
        "categories": {},
        "total_images": len(captions_data["images"]),
    }

    # Group images by category
    for category in CATEGORIES:
        dataset["categories"][category] = []

    for image_info in captions_data['images']:
        category = image_info['category']
        category_folder = f"{category.split('_')[0]}_{category.split('_', 1)[1]}"

        image_data = {
            'id': image_info['id'],
            'filename': image_info['filename'],
            'caption': image_info['caption'],
            'path': str(dataset_path / image_info['filename'])
        }

        dataset['categories'][category_folder].append(image_data)

    return dataset


def get_images_by_category(dataset_path, category):
    """
    Get all images for a specific category.

    Args:
        dataset_path: Path to dataset root directory
        category: Category name (e.g., "1_drivers_emotions")

    Returns:
        list: Image paths for that category
    """

    dataset_path = Path(dataset_path)
    category_path = dataset_path / category

    image_paths = []
    for image_file in category_path.glob(f"*{IMAGE_EXTENSION}"):
        image_paths.append(str(image_file))

    image_paths.sort()
    return image_paths


def get_all_image_paths(dataset_path):
    """
    Get all image paths from dataset across all categories.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        list: All image paths sorted
    """
    dataset_path = Path(dataset_path)

    all_paths = []
    for category in CATEGORIES:
        category_images = get_images_by_category(dataset_path, category)
        all_paths.extend(category_images)

    all_paths.sort()
    return all_paths
