"""
Caption Evaluation Script - Evaluate VLM-generated captions

This module evaluates generated captions against ground truth using standard metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4: N-gram precision metrics
- METEOR: Alignment-based metric with stemming and synonyms
- ROUGE-L: Longest common subsequence metric
- CIDEr: Consensus-based metric for image descriptions

Usage:
    python src/evaluate_captions.py --model blip
    python src/evaluate_captions.py --model git-base
    python src/evaluate_captions.py --model vit2distilgpt2
    python src/evaluate_captions.py --model swin-tiny
    python src/evaluate_captions.py --all                # evaluate all models
    python src/evaluate_captions.py --compare            # compare metrics across models

Results are saved to results/{model}/metrics.json
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Evaluation metrics libraries
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Local imports
import utils
import dataset_loader
from constants import (
    DATASET_PATH,
    AVAILABLE_MODELS,
    MODEL_RESULTS_PATHS,
    GENERATED_CAPTIONS_FILENAME,
    METRICS_FILENAME
)


# ============================================
# NLTK Data Setup
# ============================================

def ensure_nltk_data():
    """
    Download required NLTK data if not already present.

    Required for METEOR metric calculation.
    """
    try:
        nltk.data.find('wordnet')
    except LookupError:
        print("Downloading NLTK wordnet data...")
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        print("Downloading NLTK omw-1.4 data...")
        nltk.download('omw-1.4', quiet=True)


# ============================================
# Caption Loading Functions
# ============================================

def load_ground_truth(dataset_path):
    """
    Load ground truth captions from dataset.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        dict: Ground truth data from captions.json
    """
    captions_data = dataset_loader.load_captions(dataset_path)
    return captions_data


def load_generated_captions(model_name):
    """
    Load generated captions for a specific model.

    Args:
        model_name: Name of the model ("blip", "git-base", "instructblip", "phi3-vision")

    Returns:
        dict: Generated captions mapping filenames to captions

    Raises:
        FileNotFoundError: If generated captions file doesn't exist
    """
    results_dir = MODEL_RESULTS_PATHS[model_name]
    captions_path = Path(results_dir) / GENERATED_CAPTIONS_FILENAME

    if not captions_path.exists():
        raise FileNotFoundError(
            f"Generated captions not found for {model_name} at {captions_path}\n"
            f"Run inference first: python src/vlm_inference.py --model {model_name}"
        )

    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    return captions


# ============================================
# Metrics Calculation Functions
# ============================================

def calculate_bleu_scores(references, predictions):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.

    Args:
        references: List of reference caption lists (one list per image)
        predictions: List of predicted captions (one per image)

    Returns:
        dict: BLEU scores for n-grams 1-4
    """
    # Tokenize references and predictions
    tokenized_refs = [
        [ref.lower().split() for ref in ref_list]
        for ref_list in references
    ]
    tokenized_preds = [pred.lower().split() for pred in predictions]

    # Smoothing function for better handling of short captions
    smoothing = SmoothingFunction().method1

    # Calculate BLEU scores for n-grams 1-4
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        score = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=weights,
            smoothing_function=smoothing
        )
        bleu_scores[f"BLEU-{n}"] = score

    return bleu_scores


def calculate_meteor_score(references, predictions):
    """
    Calculate METEOR score (average across all images).

    Args:
        references: List of reference caption lists
        predictions: List of predicted captions

    Returns:
        float: Average METEOR score
    """
    meteor_scores = []

    for ref_list, pred in zip(references, predictions):
        # METEOR expects single reference, use first one
        # (could be extended to handle multiple references)
        ref = ref_list[0].lower()
        pred_lower = pred.lower()

        score = meteor_score([ref], pred_lower)
        meteor_scores.append(score)

    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    return avg_meteor


def calculate_rouge_l_score(references, predictions):
    """
    Calculate ROUGE-L F1 score (average across all images).

    Args:
        references: List of reference caption lists
        predictions: List of predicted captions

    Returns:
        float: Average ROUGE-L F1 score
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []

    for ref_list, pred in zip(references, predictions):
        # Use first reference
        ref = ref_list[0]

        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    return avg_rouge


def calculate_all_metrics(references, predictions):
    """
    Calculate all evaluation metrics.

    Args:
        references: List of reference caption lists
        predictions: List of predicted captions

    Returns:
        dict: All metrics with their scores
    """
    print("Calculating metrics...")

    metrics = {}

    # BLEU scores
    print("  - Computing BLEU scores...")
    bleu_scores = calculate_bleu_scores(references, predictions)
    metrics.update(bleu_scores)

    # METEOR score
    print("  - Computing METEOR score...")
    meteor = calculate_meteor_score(references, predictions)
    metrics["METEOR"] = meteor

    # ROUGE-L score
    print("  - Computing ROUGE-L score...")
    rouge_l = calculate_rouge_l_score(references, predictions)
    metrics["ROUGE-L"] = rouge_l

    print("Metrics calculation completed\n")

    return metrics


# ============================================
# Evaluation Orchestration
# ============================================

def evaluate_model(model_name):
    """
    Evaluate a single model's generated captions.

    Loads ground truth and generated captions, calculates metrics,
    and saves results to JSON.

    Args:
        model_name: Name of the model to evaluate
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()} MODEL")
    print(f"{'='*60}\n")

    # Load ground truth
    print("Loading ground truth captions...")
    ground_truth = load_ground_truth(DATASET_PATH)

    # Load generated captions
    print(f"Loading generated captions for {model_name}...")
    generated_captions = load_generated_captions(model_name)

    print(f"Found {len(generated_captions)} generated captions\n")

    # Format for evaluation
    references, predictions = utils.format_captions_for_evaluation(
        ground_truth, generated_captions
    )

    # Calculate metrics
    metrics = calculate_all_metrics(references, predictions)

    # Save metrics
    results_dir = MODEL_RESULTS_PATHS[model_name]
    metrics_path = Path(results_dir) / METRICS_FILENAME
    utils.save_metrics(metrics, str(metrics_path))

    print(f"Metrics saved to: {metrics_path}\n")

    # Display metrics
    utils.print_metrics_summary(metrics)

    print(f"{model_name.upper()} evaluation completed\n")


def evaluate_all_models():
    """
    Evaluate all available models sequentially.
    """
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)

    for model_name in AVAILABLE_MODELS:
        try:
            evaluate_model(model_name)
        except FileNotFoundError as e:
            print(f"\nSkipping {model_name}: {e}\n")
            continue

    print("="*60)
    print("ALL EVALUATIONS COMPLETED")
    print("="*60 + "\n")


# ============================================
# Comparison Functions
# ============================================

def compare_models():
    """
    Load and compare metrics across all evaluated models.

    Displays a comparison table showing all metrics side-by-side.
    """
    print("\n" + "="*60)
    print("METRICS COMPARISON ACROSS MODELS")
    print("="*60 + "\n")

    # Load metrics for all models
    all_metrics = {}
    for model_name in AVAILABLE_MODELS:
        results_dir = MODEL_RESULTS_PATHS[model_name]
        metrics_path = Path(results_dir) / METRICS_FILENAME

        if metrics_path.exists():
            metrics = utils.load_metrics(str(metrics_path))
            all_metrics[model_name] = metrics
        else:
            print(f"Warning: Metrics not found for {model_name}")

    if not all_metrics:
        print("No metrics found. Run evaluation first.")
        return

    # Get all metric names
    metric_names = list(next(iter(all_metrics.values())).keys())

    # Print header
    model_names = list(all_metrics.keys())
    header = f"{'Metric':<15}"
    for model in model_names:
        header += f"{model.upper():<15}"
    print(header)
    print("-" * len(header))

    # Print each metric
    for metric_name in metric_names:
        row = f"{metric_name:<15}"
        for model in model_names:
            value = all_metrics[model].get(metric_name, 0.0)
            if isinstance(value, float):
                row += f"{value:<15.4f}"
            else:
                row += f"{value:<15}"
        print(row)

    print("\n" + "="*60 + "\n")

    # Find best model per metric
    print("BEST MODEL PER METRIC:")
    print("-" * 40)
    for metric_name in metric_names:
        best_model = max(
            model_names,
            key=lambda m: all_metrics[m].get(metric_name, 0.0)
        )
        best_score = all_metrics[best_model][metric_name]
        print(f"{metric_name:<15}: {best_model.upper():<10} ({best_score:.4f})")

    print("="*60 + "\n")


# ============================================
# CLI Functions
# ============================================

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate VLM-generated captions against ground truth"
    )

    # Create mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        help="Evaluate specific model"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models"
    )

    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare metrics across all evaluated models"
    )

    args = parser.parse_args()
    return args


def validate_arguments(args):
    """
    Validate parsed arguments and check prerequisites.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check dataset exists (only needed if not comparing)
    if not args.compare:
        dataset_path = Path(DATASET_PATH)
        if not dataset_path.exists():
            print(f"Error: Dataset not found at {DATASET_PATH}")
            sys.exit(1)

        captions_file = dataset_path / "captions.json"
        if not captions_file.exists():
            print(f"Error: Ground truth captions not found at {captions_file}")
            sys.exit(1)

        print("Dataset validation passed\n")


def main():
    """
    Main entry point for caption evaluation.
    """
    # Ensure NLTK data is available
    ensure_nltk_data()

    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)

    # Execute based on mode
    if args.compare:
        compare_models()
    elif args.all:
        evaluate_all_models()
    else:
        evaluate_model(args.model)


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    main()
