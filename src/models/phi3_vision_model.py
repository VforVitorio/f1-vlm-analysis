"""
Phi-3-Vision model wrapper for image captioning with prompt tuning.

Model: microsoft/Phi-3-vision-128k-instruct
Size: ~3.8B parameters (4-bit quantized to ~2.5GB)
VRAM: ~2.5-3GB with 4-bit quantization
Speed: ~1-2 seconds/image

Phi-3-Vision is Microsoft's efficient multimodal model with strong instruction-following
capabilities and excellent performance on vision-language tasks.
"""

from utils import load_image, clear_gpu_cache, get_device_from_arg
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import sys

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))


# ============================================
# Constants
# ============================================
MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct"
DEFAULT_MAX_LENGTH = 50
DEFAULT_PROMPT = "Describe this image."


# ============================================
# Phi3VisionModel Class
# ============================================
class Phi3VisionModel:
    """
    Wrapper for Phi-3-Vision multimodal model.

    Phi-3-Vision is optimized for efficiency with strong instruction-following capabilities.
    Uses 4-bit quantization to fit in limited VRAM.
    """

    def __init__(self, device=None, custom_prompt=None):
        """
        Initialize Phi-3-Vision model and processor.

        Loads model with 4-bit quantization for memory efficiency.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
            custom_prompt: Custom instruction prompt for captioning. Uses default if None.
        """
        # Get device using utils function
        self.device = get_device_from_arg(device)
        self.prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

        print(f"Loading Phi-3-Vision model on {self.device}...")
        print(f"Using prompt: '{self.prompt}'")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Configure 4-bit quantization for VRAM efficiency
        if self.device.type == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                _attn_implementation='eager'  # Use eager attention for compatibility
            )
        else:
            # CPU inference without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="cpu",
                trust_remote_code=True
            )

        # Set to evaluation mode
        self.model.eval()

        print("Phi-3-Vision model loaded successfully")

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

        # Prepare messages in Phi-3 format
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]

        # Apply chat template
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs
        inputs = self.processor(
            prompt_text,
            [image],
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = inputs.to(self.device)

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        # Decode caption (remove prompt from output)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        caption = self.processor.decode(
            generated_ids,
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
def create_phi3_vision_model(device=None, custom_prompt=None):
    """
    Create Phi-3-Vision model instance.

    Factory function for consistent model creation across the pipeline.

    Args:
        device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        custom_prompt: Custom instruction prompt for captioning.

    Returns:
        Phi3VisionModel: Initialized Phi-3-Vision model
    """
    return Phi3VisionModel(device=device, custom_prompt=custom_prompt)
