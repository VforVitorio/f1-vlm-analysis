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
