# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset comparing three ultra-lightweight architectures optimized for GTX 1050 (4GB VRAM).

## Models Used

### 1. BLIP-base
- **Model**: `Salesforce/blip-image-captioning-base`
- **Architecture**: ViT + BERT decoder (VisionEncoderDecoder)
- **VRAM**: ~1.5GB | **Speed**: ~1-2s/image
- **Quality**: ✅ Good F1 captions

### 2. GIT-base
- **Model**: `microsoft/git-base`
- **Architecture**: ViT + GPT-2 decoder (VisionEncoderDecoder)
- **VRAM**: ~1-2GB | **Speed**: ~1s/image
- **Quality**: ✅ Good F1 captions

### 3. Swin-Tiny-DistilGPT2 ⚠️
- **Model**: `yesidcanoc/image-captioning-swin-tiny-distilgpt2`
- **Architecture**: Swin-Tiny + DistilGPT2 decoder (VisionEncoderDecoder)
- **VRAM**: <1GB | **Speed**: <1s/image
- **Quality**: ❌ Poor (generic COCO captions, not F1-relevant)
- **Purpose**: Negative control - demonstrates checkpoint quality matters more than architecture

## Project Structure

```
practica_3_vlm/
├── Dockerfile                # Docker configuration with CUDA support
├── Makefile                  # Build and execution recipes
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── models/              # VLM model implementations
│   │   ├── __init__.py
│   │   ├── blip_model.py    # BLIP-base captioner class
│   │   ├── git_base_model.py  # GIT-base captioner class
│   │   └── swin_tiny_model.py   # Swin-Tiny-DistilGPT2 captioner class
│   ├── vlm_inference.py     # Main orchestrator (handles --model arg)
│   ├── dataset_loader.py    # Load F1 dataset with categories
│   ├── evaluation.py        # Metrics computation (BLEU, ROUGE, etc.)
│   ├── comparison.py        # Cross-model comparative analysis
│   └── utils.py             # Helper functions
├── dataset/                  # F1 dataset from practice 2
│   ├── 1_drivers_emotions/  # Drivers celebrating, frustrated
│   │   ├── driver_01.jpg
│   │   └── ...
│   ├── 2_pit_stops/         # Mechanics, tire changes
│   │   ├── pitstop_01.jpg
│   │   └── ...
│   ├── 3_cars_tracks_moments/  # Racing action, crashes
│   │   ├── car_01.jpg
│   │   └── ...
│   ├── 4_strategy_data/     # Engineers, telemetry screens
│   │   ├── strategy_01.jpg
│   │   └── ...
│   └── captions.json        # Ground truth captions
└── results/
    ├── blip/                # BLIP-base results
    │   ├── generated_captions.json
    │   └── metrics.json
    ├── git-base/            # GIT-base results
    │   ├── generated_captions.json
    │   └── metrics.json
    ├── swin-tiny/           # Swin-Tiny-DistilGPT2 results
    │   ├── generated_captions.json
    │   └── metrics.json
    └── comparison/          # Cross-model comparison
        └── comparative_analysis.json
```

## Dataset Structure

> [!WARNING]
>
> **Dataset Not Included**
>
> The F1 image dataset is **NOT uploaded to this repository** due to size constraints.
>
> If you need access to the dataset for reproduction purposes, please contact me through the links available in my GitHub bio.

### F1 Dataset Overview

- **Total images** : 20
- **Categories** : 4 thematic groups
- **Language** : English captions
- **Source** : From Practice 2 (CLIP analysis)

### Dataset Categories

1. **Drivers and Emotions** (5 images)
   - Podium celebrations, frustration moments
   - Focus: Facial expressions, body language, team context
2. **Pit Stops and Mechanics** (5 images)
   - Tire changes, technical work in pit lane
   - Focus: Coordination, precision, racing conditions
3. **Cars and Track Moments** (5 images)
   - Racing action, overtakes, crashes, different circuits
   - Focus: Vehicle identification, track features, weather conditions
4. **Strategy and Data** (5 images)
   - Engineers monitoring telemetry, team discussions
   - Focus: Abstract technical content, screens with data

### captions.json Format

```json
{
  "images": [
    {
      "id": 1,
      "category": "drivers_emotions",
      "filename": "1_drivers_emotions/driver_01.jpg",
      "caption": "Driver in AlphaTauri suit sitting with hand on face and trophy and champagne beside him"
    },
    {
      "id": 6,
      "category": "pit_stops_mechanics",
      "filename": "2_pit_stops/pitstop_01.jpg",
      "caption": "McLaren mechanics in orange suits coordinating during pit stop at night"
    }
  ]
}
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA (tested on GTX 1050, 4GB VRAM)
- ~10GB disk space for model weights

## Usage

### Quick Start

```bash
make build        # Build Docker image
make run-all      # Generate captions with all models (<1 min)
make evaluate     # Compute metrics
make compare      # Generate comparative analysis
```

### Individual Models

```bash
make run-blip       # BLIP-base only
make run-git-base   # GIT-base only
make run-swin-tiny  # Swin-Tiny only
```

### Other Commands

```bash
make shell         # Interactive container shell
make gpu-info      # Check GPU/VRAM status
make clean         # Remove Docker image
make clean-results # Delete generated results
```

## Evaluation Metrics

- **BLEU-1, BLEU-4**: N-gram precision
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Alignment-based metric
- **CIDEr**: Consensus-based metric

Results include overall scores, per-category breakdowns, and best/worst performing images.

## Notes

This project uses **VisionEncoderDecoder models** (BLIP, GIT, Swin-Tiny) instead of modern LLM-based VLMs due to **VRAM constraints on GTX 1050 (4GB)**. First-Gen VLMs offer <2GB VRAM usage, instant loading (<10s), and fast inference (<2s/image), but cannot follow instructions or support prompt tuning. Modern LLM-based VLMs (Qwen2-VL, Phi-3-Vision, InstructBLIP) require >4GB VRAM even with 4-bit quantization, suffer from extremely slow loading times (16min-1.5hr timeouts), and produce OOM errors during generation. While smaller modern VLMs exist (PaliGemma-3B, SmolVLM-2B, MobileVLM-1.7B), they still require ~3.5-4GB minimum (LLM + vision encoder + activation overhead), leaving no safety margin on 4GB hardware. Trade-off: we sacrifice instruction-following capabilities for 100% reliability and speed on limited hardware.

## VLM Architecture Overview

This project uses **first-generation VLMs** (VisionEncoderDecoder architecture, 2021-2022 era) optimized for low VRAM environments. These differ fundamentally from modern LLM-based multimodal models:

### 🔹 First-Gen VLMs (Used in This Project)

**Architecture**: `Vision Encoder → Cross-Attention → Language Decoder`

- **Training**: Image-to-text on COCO with simple captions
- **Capability**: Generate descriptions from images
- **Prompting**: Limited - only short semantic primes (2-5 words)
- **VRAM**: Very low (<2GB), perfect for GTX 1050
- **Examples**: BLIP, GIT, Swin-Tiny, ViT-GPT2

**Limitations**: Cannot follow complex instructions, no reasoning capabilities, prompts must be minimal.

### 🔹 Modern LLM-based VLMs (Failed on 4GB VRAM)

**Architecture**: `Vision Encoder → Projection → LLM (Llama/Qwen/Phi)`

- **Training**: Instruction tuning with multimodal dialogues
- **Capability**: Follow instructions, reasoning, Q&A, chat
- **Prompting**: Full natural language instructions
- **VRAM**: High (>4GB even with quantization)
- **Examples**: Qwen2-VL, Phi-3-Vision, InstructBLIP, MiniCPM-V

**Why They Failed**: Models like Qwen2-VL (OOM at 4.27GB), Phi-3-Vision (shard loading timeout), and InstructBLIP (failed to load after 1.5 hours) require significantly more VRAM and computational resources than available on GTX 1050.
