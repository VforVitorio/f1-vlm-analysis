"""
MiniCPM-V model wrapper for image captioning.

Model: openbmb/MiniCPM-Llama3-V-2_5
Size: ~8GB (4-bit quantization)
VRAM: ~5-6GB (with 4-bit quantization)
Speed: ~3-5 seconds/image

MiniCPM-V is a compact multimodal LLM with strong vision-language capabilities.
Uses 4-bit quantization for efficiency.
"""

import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_image, clear_gpu_cache, get_device_from_arg


# ============================================
# Constants
# ============================================
MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"
DEFAULT_MAX_LENGTH = 50
DEFAULT_PROMPT = "Describe this image in one sentence."


# ============================================
# MiniCPMModel Class
# ============================================
class MiniCPMModel:
    """
    Wrapper for MiniCPM-V multimodal language model.

    MiniCPM-V uses a chat-based interface for vision-language tasks.
    Supports 4-bit quantization for memory efficiency.
    """

    def __init__(self, device=None):
        """
        Initialize MiniCPM-V model and tokenizer.

        Loads model with 4-bit quantization for efficiency.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)

        print(f"Loading MiniCPM-V model on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model with 4-bit quantization if on CUDA
        if self.device.type == 'cuda':
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
        else:
            # CPU inference without quantization
            self.model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                device_map='cpu'
            )

        # Set to evaluation mode
        self.model.eval()

        print("MiniCPM-V model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH):
        """
        Generate caption for a single image.

        Uses chat-based interface with a descriptive prompt.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length (not directly used by MiniCPM API)

        Returns:
            str: Generated caption
        """
        # Load image using utils function
        image = load_image(image_path)

        # Prepare messages in chat format
        # MiniCPM-V expects a conversation-style input
        messages = [
            {
                'role': 'user',
                'content': DEFAULT_PROMPT
            }
        ]

        # Generate caption using MiniCPM's chat interface
        # The model handles image and text jointly
        with torch.no_grad():
            response = self.model.chat(
                image=image,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7
            )

        # Extract text from response
        # Response is typically a string
        caption = response.strip()

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
def create_minicpm_model(device=None):
    """
    Create MiniCPM-V model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.

    Returns:
        MiniCPMModel: Initialized MiniCPM-V model
    """
    return MiniCPMModel(device=device)
