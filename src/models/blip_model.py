"""
BLIP-base model wrapper for image captioning.

Model: Salesforce/blip-image-captioning-base
Size: ~450MB (FP16)
VRAM: ~1.5GB
Speed: ~1-2 seconds/image
"""

import torch
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration


# ============================================
# Constants
# ============================================
MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_MAX_LENGTH = 50
DEFAULT_BATCH_SIZE = 4


# ============================================
# BLIPModel Class
# ============================================
class BLIPModel:
    """
    Wrapper for BLIP-base image captioning model.
    """

    def __init__(self, device=None):
        """
        Initialize BLIP model and processor.

        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detect if None.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading BLIP model on {self.device}...")

        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        print("BLIP model loaded successfully")

    def generate_caption(self, image_path, max_length=DEFAULT_MAX_LENGTH):
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            max_length: Maximum caption length in tokens

        Returns:
            str: Generated caption
        """
        # Load image
        image = self._load_image(image_path)

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to device
        pixel_values = inputs['pixel_values'].to(self.device)

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )

        # Decode caption
        caption = self.processor.decode(
            output_ids[0], skip_special_tokens=True)

        return caption

    def generate_captions_batch(self, image_paths, max_length=DEFAULT_MAX_LENGTH,
                                batch_size=DEFAULT_BATCH_SIZE):
        """
        Generate captions for multiple images in batches.

        Args:
            image_paths: List of image paths
            max_length: Maximum caption length in tokens
            batch_size: Number of images per batch

        Returns:
            list: Generated captions
        """
        all_captions = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Load images for this batch
            batch_images = []
            for path in batch_paths:
                image = self._load_image(path)
                batch_images.append(image)

            # Preprocess batch
            inputs = self.processor(
                images=batch_images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(self.device)

            # Generate captions
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode each caption
            for output_id in output_ids:
                caption = self.processor.decode(
                    output_id, skip_special_tokens=True)
                all_captions.append(caption)

        return all_captions

    def _load_image(self, image_path):
        """
        Load image from path.

        Args:
            image_path: Path to image file

        Returns:
            PIL.Image: Loaded image in RGB format
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def clear_cache(self):
        """
        Clear GPU cache to free up VRAM.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================
# Factory Function
# ============================================
def create_blip_model(device=None):
    """
    Create BLIP model instance.

    Args:
        device: torch device

    Returns:
        BLIPModel: Initialized BLIP model
    """
    return BLIPModel(device=device)
