"""
VLM Inference - Main orchestrator for image captioning

This module handles the complete inference pipeline:
- CLI argument parsing (--model blip/git-base/instructblip/phi3-vision or --all)
- Model loading and initialization
- Dataset processing and caption generation
- Results saving to JSON format

Usage:
    python src/vlm_inference.py --model blip
    python src/vlm_inference.py --model git-base
    python src/vlm_inference.py --model instructblip
    python src/vlm_inference.py --model phi3-vision
    python src/vlm_inference.py --all

The module coordinates between model implementations, dataset loading,
and utility functions to generate captions for the F1 dataset.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Local imports
from models.blip_model import create_blip_model
from models.git_base_model import create_git_base_model
from models.instructblip_model import create_instructblip_model
from models.phi3_vision_model import create_phi3_vision_model

import utils
import dataset_loader
from constants import (
    DATASET_PATH,
    AVAILABLE_MODELS,
    MODEL_RESULTS_PATHS,
    GENERATED_CAPTIONS_FILENAME
)

# ============================================
# Model Management Functions
# ============================================


def create_model(model_name):
    """
    Factory function to create model instance based on name.

    Args:
        model_name: Name of the model ("blip", "git-base", "instructblip", "phi3-vision")
    Returns:
        Model instance with generate_caption() method

    Raises:
        ValueError: If model_name is not supported
    """

    if model_name == "blip":
        return create_blip_model()
    elif model_name == "git-base":
        return create_git_base_model()
    elif model_name == "instructblip":
        return create_instructblip_model()
    elif model_name == "phi3-vision":
        return create_phi3_vision_model()
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")


def initialize_model(model_name):
    """
    Initialize model with appropriate device configuration.

    Detects available device (CUDA/CPU) and creates model instance.

    Args:
        model_name: Name of the model to initialize

    Returns:
        Initialized model instance ready for inference
    """
    device = utils.get_device()

    print(f"\n{'='*60}")
    print(f"Initializing {model_name.upper()} model...")
    print(f"{'='*60}\n")

    model = create_model(model_name)

    print(f"\n{model_name.upper()} model ready for inference\n")

    return model

# ============================================
# Path Handling Functions
# ============================================


def get_relative_path(image_path, dataset_path):
    """
    Convert absolute image path to relative path for JSON storage.

    Extracts the relative path from dataset root to maintain consistency
    with captions.json format.

    Args:
        image_path: Absolute path to image file
        dataset_path: Absolute path to dataset root directory

    Returns:
        str: Relative path from dataset root (e.g., "1_drivers_emotions/driver_01.jpg")

    Example:
        >>> get_relative_path("/app/dataset/1_drivers_emotions/driver_01.jpg", "/app/dataset")
        "1_drivers_emotions/driver_01.jpg"
    """
    image_path = Path(image_path)
    dataset_path = Path(dataset_path)

    relative_path = image_path.relative_to(dataset_path)

    return str(relative_path)

# ============================================
# Caption Generation Functions
# ============================================


def process_all_images(model, image_paths, dataset_path):
    """
    Process all images in dataset and generate caption.

    Iterates through image paths, generate captions and display progress.
    Returns results in format compatible with ground truth JSON structure


    Args:
        model: VLM model instance
        image_paths: List of absolute paths to images
        dataset_path: Root path of dataset (for relative path conversion)

    Returns:
        dict: Mapping of relative image paths to generated captions
              Format: {"1_drivers_emotions/driver_01.jpg": "caption text", ...}
    """
    captions = {}
    total_images = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        # Get relative path for JSON storage
        relative_path = get_relative_path(image_path, dataset_path)

        # Display progress
        utils.print_progress(idx, total_images, relative_path)

        # Generate caption
        caption = model.generate_caption(image_path)

        # Store result
        captions[relative_path] = caption

    print(f"\nCompleted: {total_images} images processed\n")

    return captions


# ============================================
# Orchestration Functions
# ============================================

def run_single_model(model_name):
    """
    Execute complete inference pipeline for a single model.

    Performs the full workflow:
    1. Initialize model
    2. Load dataset image paths
    3. Generate captions for all images
    4. Save results to JSON
    5. Cleanup GPU memory

    Args:
        model_name: Name of the model to run
    """
    # Initialize model
    model = initialize_model(model_name)

    # Get dataset images
    image_paths = dataset_loader.get_all_image_paths(DATASET_PATH)
    print(f"Found {len(image_paths)} images in dataset\n")

    # Generate captions
    captions = process_all_images(model, image_paths, DATASET_PATH)

    # Prepare output path
    output_dir = MODEL_RESULTS_PATHS[model_name]
    output_path = Path(output_dir) / GENERATED_CAPTIONS_FILENAME

    # Save results (ensure_output_dir is called internally)
    utils.save_captions_json(captions, str(output_path))

    print(f"Results saved to: {output_path}")

    # Cleanup GPU memory (clear_gpu_cache is called internally)
    model.clear_cache()

    print(f"\n{model_name.upper()} inference completed successfully\n")


def run_all_models():
    """
    Execute inference pipeline for all available models sequentially.

    Processes each model in AVAILABLE_MODELS list, ensuring proper
    cleanup between models to manage VRAM efficiently.
    """
    print("\n" + "="*60)
    print("RUNNING ALL MODELS SEQUENTIALLY")
    print("="*60 + "\n")

    for model_name in AVAILABLE_MODELS:
        run_single_model(model_name)

    print("\n" + "="*60)
    print("ALL MODELS COMPLETED")
    print("="*60 + "\n")


# ============================================
# CLI Functions
# ============================================

def parse_arguments():
    """
    Parse command line arguments.

    Supports two modes:
    - Single model: --model {blip, git-base, instructblip, phi3-vision}
    - All models: --all

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="VLM Image Captioning - Generate captions for F1 dataset"
    )

    # Create mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        help="Run inference with specific model"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run inference with all models sequentially"
    )

    args = parser.parse_args()
    return args


def validate_arguments(args):
    """
    Validate parsed arguments and check prerequisites.

    Verifies:
    - Dataset path exists
    - Model name is valid (if specified)

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check dataset exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    # Check captions.json exists
    captions_file = dataset_path / "captions.json"
    if not captions_file.exists():
        print(f"Error: captions.json not found at {captions_file}")
        sys.exit(1)

    print("Dataset validation passed\n")


def main():
    """
    Main entry point for VLM inference pipeline.

    Parses CLI arguments and executes appropriate inference mode.
    """
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)

    # Execute based on mode
    if args.all:
        run_all_models()
    else:
        run_single_model(args.model)


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    main()
