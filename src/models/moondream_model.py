"""
Moondream2 model wrapper for image captioning.

Model: vikhyatk/moondream2
Size: ~2GB
VRAM: ~3-4GB
Speed: ~2-3 seconds/image

Moondream2 is a compact vision-language model optimized for efficiency.
Uses a question-answering approach for caption generation.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_image, clear_gpu_cache, get_device_from_arg


# ============================================
# Constants
# ============================================
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"  # Stable version
DEFAULT_MAX_LENGTH = 50
DEFAULT_PROMPT = "Describe this image."


# ============================================
# MoondreamModel Class
# ============================================
class MoondreamModel:
    """
    Wrapper for Moondream2 vision-language model.

    Moondream uses a question-answering approach where you ask
    questions about images and the model generates answers.
    """

    def __init__(self, device=None):
        """
        Initialize Moondream2 model and tokenizer.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)

        print(f"Loading Moondream2 model on {self.device}...")

        # Load model with specific revision for stability
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print("Moondream2 model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH):
        """
        Generate caption for a single image.

        Uses the question-answering approach with a descriptive prompt.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length (not directly used by Moondream API)

        Returns:
            str: Generated caption
        """
        # Load image using utils function
        image = load_image(image_path)

        # Encode image into embeddings
        image_embeds = self.model.encode_image(image)

        # Generate caption by answering the descriptive question
        # Note: Moondream's answer_question handles generation internally
        with torch.no_grad():
            caption = self.model.answer_question(
                image_embeds,
                DEFAULT_PROMPT,
                self.tokenizer
            )

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
def create_moondream_model(device=None):
    """
    Create Moondream2 model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.

    Returns:
        MoondreamModel: Initialized Moondream2 model
    """
    return MoondreamModel(device=device)
