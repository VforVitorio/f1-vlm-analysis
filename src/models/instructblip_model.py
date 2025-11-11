"""
InstructBLIP model wrapper for image captioning with prompt tuning.

Model: Salesforce/instructblip-vicuna-7b
Size: ~7B parameters (4-bit quantized to ~3.5GB)
VRAM: ~3-3.5GB with 4-bit quantization
Speed: ~2-3 seconds/image

InstructBLIP extends BLIP-2 with instruction tuning, allowing custom prompts
for more controlled and detailed image captioning.
"""

from utils import load_image, clear_gpu_cache, get_device_from_arg
import torch
from pathlib import Path
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))


# ============================================
# Constants
# ============================================
MODEL_NAME = "Salesforce/instructblip-vicuna-7b"
DEFAULT_MAX_LENGTH = 50
DEFAULT_PROMPT = "Describe this image in detail."


# ============================================
# InstructBLIPModel Class
# ============================================
class InstructBLIPModel:
    """
    Wrapper for InstructBLIP vision-language model.

    InstructBLIP allows instruction-based image captioning with customizable prompts.
    Uses 4-bit quantization to fit in limited VRAM.
    """

    def __init__(self, device=None, custom_prompt=None):
        """
        Initialize InstructBLIP model and processor.

        Loads model with 4-bit quantization for memory efficiency.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
            custom_prompt: Custom instruction prompt for captioning. Uses default if None.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)
        self.prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

        print(f"Loading InstructBLIP model on {self.device}...")
        print(f"Using prompt: '{self.prompt}'")

        # Load processor
        self.processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)

        # Configure 4-bit quantization for VRAM efficiency
        if self.device.type == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # CPU inference without quantization
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                device_map="cpu"
            )

        # Set to evaluation mode
        self.model.eval()

        print("InstructBLIP model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH, custom_prompt=None):
        """
        Generate caption for a single image using instruction prompt.

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
        prompt = custom_prompt if custom_prompt else self.prompt

        # Prepare inputs with prompt
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = inputs.to(self.device)

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=5,
                early_stopping=True
            )

        # Decode caption
        caption = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0].strip()

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
def create_instructblip_model(device=None, custom_prompt=None):
    """
    Create InstructBLIP model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        custom_prompt: Custom instruction prompt for captioning.

    Returns:
        InstructBLIPModel: Initialized InstructBLIP model
    """
    return InstructBLIPModel(device=device, custom_prompt=custom_prompt)
