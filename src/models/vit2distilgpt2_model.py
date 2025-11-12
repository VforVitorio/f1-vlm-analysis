"""
ViT2DistilGPT2 model wrapper for image captioning with prompt tuning support.

Model: sachin/vit2distilgpt2
Size: ~0.2B parameters (~200MB)
VRAM: <1GB (FP16)
Speed: <1 second/image

VisionEncoderDecoder model with ViT-base encoder + DistilGPT2 decoder.
Ultra-lightweight and fast, supports custom prompts through generation parameters.
"""

from utils import load_image, clear_gpu_cache, get_device_from_arg
import torch
from pathlib import Path
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))


# ============================================
# Constants
# ============================================
MODEL_NAME = "sachin/vit2distilgpt2"
DEFAULT_MAX_LENGTH = 50
DEFAULT_PROMPT = "Describe this image."


# ============================================
# ViT2DistilGPT2Model Class
# ============================================
class ViT2DistilGPT2Model:
    """
    Wrapper for ViT2DistilGPT2 VisionEncoderDecoder model.

    Ultra-lightweight model combining ViT encoder with DistilGPT2 decoder.
    Supports prompt tuning through custom generation prefixes.
    """

    def __init__(self, device=None, custom_prompt=None):
        """
        Initialize ViT2DistilGPT2 model, processor, and tokenizer.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
            custom_prompt: Custom prompt prefix for caption generation.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)
        self.prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

        print(f"Loading ViT2DistilGPT2 model on {self.device}...")
        print(f"Using prompt prefix: '{self.prompt}'")

        # Load model
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)

        # Load image processor from ViT base (the encoder model)
        # sachin/vit2distilgpt2 doesn't have preprocessor_config.json
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        # Load tokenizer (DistilGPT2)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Configure tokenizer special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set to evaluation mode
        self.model.eval()

        print("ViT2DistilGPT2 model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH, custom_prompt=None):
        """
        Generate caption for a single image with optional prompt tuning.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length in tokens
            custom_prompt: Override the default prompt for this specific image

        Returns:
            str: Generated caption
        """
        # Load image using utils function
        image = load_image(image_path)

        # Use custom prompt if provided, otherwise use instance prompt
        prompt_text = custom_prompt if custom_prompt else self.prompt

        # Process image
        pixel_values = self.image_processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Tokenize prompt as decoder input prefix (prompt tuning)
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        # Generate caption with prompt prefix
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=prompt_ids,  # Use prompt as prefix
                max_length=max_length + prompt_ids.shape[1],
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode caption (remove prompt prefix from output)
        caption = self.tokenizer.decode(
            outputs[0][prompt_ids.shape[1]:],  # Skip prompt tokens
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
def create_vit2distilgpt2_model(device=None, custom_prompt=None):
    """
    Create ViT2DistilGPT2 model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        custom_prompt: Custom prompt prefix for caption generation.

    Returns:
        ViT2DistilGPT2Model: Initialized ViT2DistilGPT2 model
    """
    return ViT2DistilGPT2Model(device=device, custom_prompt=custom_prompt)
