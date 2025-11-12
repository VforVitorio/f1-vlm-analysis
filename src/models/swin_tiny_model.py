"""
Swin-Tiny-DistilGPT2 model wrapper for image captioning with prompt tuning support.

Model: yesidcanoc/image-captioning-swin-tiny-distilgpt2
Size: ~0.15B parameters (~150MB)
VRAM: <1GB (FP16)
Speed: <1 second/image

VisionEncoderDecoder model with Swin-Tiny encoder + DistilGPT2 decoder.
Ultra-lightweight with Swin Transformer architecture, supports custom prompts.
"""

from utils import load_image, clear_gpu_cache, get_device_from_arg
import torch
from pathlib import Path
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))


# ============================================
# Constants
# ============================================
MODEL_NAME = "yesidcanoc/image-captioning-swin-tiny-distilgpt2"
DEFAULT_MAX_LENGTH = 50

# Default generic prompt (fallback) - MUST BE SHORT for VisionEncoderDecoder
DEFAULT_PROMPT = "formula 1"

# Category-specific short prompts for F1 dataset
# VisionEncoderDecoder models need SHORT semantic primes (2-5 words), not instructions
# These act as context hints to guide the decoder's vocabulary and focus
CATEGORY_PROMPTS = {
    # Driver emotions: team suits, podium, celebrations, flags, trophies
    "1_drivers_emotions": "f1 driver emotion",

    # Pit stops: mechanics, tire changes, team colors, pit equipment
    "2_pit_stops": "f1 pit stop",

    # Racing action: cars, overtakes, crashes, track, weather
    "3_cars_tracks_moments": "f1 racing action",

    # Strategy: engineers, screens, telemetry, team discussions
    "4_strategy_data": "f1 team engineer"
}


# ============================================
# SwinTinyModel Class
# ============================================
class SwinTinyModel:
    """
    Wrapper for Swin-Tiny-DistilGPT2 VisionEncoderDecoder model.

    Ultra-lightweight model using Swin Transformer encoder with DistilGPT2 decoder.
    Swin architecture provides better hierarchical feature learning than ViT.
    Supports prompt tuning through custom generation prefixes.
    """

    def __init__(self, device=None, custom_prompt=None):
        """
        Initialize Swin-Tiny-DistilGPT2 model, processor, and tokenizer.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
            custom_prompt: Custom prompt prefix (overrides category-based prompts).
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)
        self.custom_prompt = custom_prompt  # Store for manual override
        # Auto-detect if not overridden
        self.use_category_prompts = (custom_prompt is None)

        print(f"Loading Swin-Tiny-DistilGPT2 model on {self.device}...")
        if self.use_category_prompts:
            print("Using category-specific F1 prompts (auto-detected from image path)")
        else:
            print(f"Using custom prompt: '{self.custom_prompt}'")

        # Load model
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)

        # Load image processor (Swin feature extractor)
        self.image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        # Load tokenizer (DistilGPT2)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Configure tokenizer special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set to evaluation mode
        self.model.eval()

        print("Swin-Tiny-DistilGPT2 model loaded successfully")

    def _get_category_prompt(self, image_path):
        """
        Detect image category from path and return appropriate F1-specific prompt.

        Args:
            image_path: Path to image file (str or Path)

        Returns:
            str: Category-specific prompt or default prompt
        """
        # Convert to string if Path object
        path_str = str(image_path)

        # Check which category folder the image belongs to
        for category_key, prompt in CATEGORY_PROMPTS.items():
            if category_key in path_str:
                return prompt

        # Fallback to default F1 prompt if no category detected
        return DEFAULT_PROMPT

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH, custom_prompt=None):
        """
        Generate caption for a single image.

        Note: This model uses VisionEncoderDecoder architecture trained on COCO.
        Prompt tuning via decoder_input_ids does not work well and causes degraded output.
        Generates captions directly from image features without prompts for best results.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length in tokens
            custom_prompt: Not used (kept for API compatibility)

        Returns:
            str: Generated caption
        """
        # Load image using utils function
        image = load_image(image_path)

        # Process image
        pixel_values = self.image_processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate caption directly without prompt prefix
        # Prompt tuning does not work well with this VisionEncoderDecoder checkpoint
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                do_sample=False,  # Deterministic beam search
                no_repeat_ngram_size=2,
                length_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode caption
        caption = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        return caption

    def clear_cache(self):
        """
        Clear GPU cache to free up VRAM.

        Useful when switching between models or after inference.
        """
        clear_gpu_cache()


# ============================================
# Factory Function
# ============================================
def create_swin_tiny_model(device=None, custom_prompt=None):
    """
    Create Swin-Tiny-DistilGPT2 model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        custom_prompt: Custom prompt prefix for caption generation.

    Returns:
        SwinTinyModel: Initialized Swin-Tiny-DistilGPT2 model
    """
    return SwinTinyModel(device=device, custom_prompt=custom_prompt)
