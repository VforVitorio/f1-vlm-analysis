"""
Project Constants - Shared configuration across modules

This module centralizes all constants used across the VLM inference
and evaluation pipeline to avoid duplication and ensure consistency.
"""

# ============================================
# Path Constants
# ============================================
DATASET_PATH = "dataset"
RESULTS_BASE_PATH = "results"

# ============================================
# Model Configuration
# ============================================
AVAILABLE_MODELS = ["blip", "git-base", "swin-tiny"]

MODEL_RESULTS_PATHS = {
    "blip": "results/blip",
    "git-base": "results/git-base",
    "swin-tiny": "results/swin-tiny"
}

# ============================================
# File Names
# ============================================
CAPTIONS_FILENAME = "captions.json"
GENERATED_CAPTIONS_FILENAME = "generated_captions.json"
METRICS_FILENAME = "metrics.json"
