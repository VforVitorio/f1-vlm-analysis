"""
GIT-base model wrapper for image captioning.

Model: microsoft/git-base
Size: ~350MB
VRAM: ~1-2GB (perfect for GTX 1050)
Speed: ~1 second/image

GIT (GenerativeImage2Text) is a lightweight and efficient vision-language model
designed for image captioning tasks with excellent speed and quality balance.
"""

from utils import load_image, clear_gpu_cache, get_device_from_arg
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))


# ============================================
# Constants
# ============================================
MODEL_NAME = "microsoft/git-base"
DEFAULT_MAX_LENGTH = 50


# ============================================
# GitBaseModel Class
# ============================================
class GitBaseModel:
    """
    Wrapper for Microsoft GIT-base vision-language model.

    GIT (Generative Image-to-text) is designed for efficient image captioning
    with a good balance between speed and quality.
    """

    def __init__(self, device=None):
        """
        Initialize GIT-base model and processor.

        Loads model with FP16 precision on CUDA for efficiency.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)

        print(f"Loading GIT-base model on {self.device}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

        # Load model with FP16 if on CUDA
        if self.device.type == 'cuda':
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16
            )
        else:
            # CPU inference without FP16
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print("GIT-base model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH):
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length in tokens

        Returns:
            str: Generated caption
        """
        # Load image using utils function
        image = load_image(image_path)

        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to device
        inputs = inputs.to(self.device)

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode caption
        caption = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return caption.strip()

    def clear_cache(self):
        """
        Clear GPU cache to free up VRAM.

        Useful when switching between models or after inference.
        """
        clear_gpu_cache()


# ============================================
# Factory Function
# ============================================
def create_git_base_model(device=None):
    """
    Create GIT-base model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.

    Returns:
        GitBaseModel: Initialized GIT-base model
    """
    return GitBaseModel(device=device)
