# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset comparing three ultra-lightweight architectures.

## Models Used

This project implements and compares **three VLM architectures** optimized for 4GB VRAM (GTX 1050):

### 1. BLIP-base (Baseline - No Prompts)

- **Model** : `Salesforce/blip-image-captioning-base`
- **Size** : ~450MB (FP16)
- **VRAM** : ~1.5GB
- **Purpose** : Establishes baseline performance with proven architecture
- **Speed** : ~1-2 seconds/image
- **Prompts** : ❌ No - Direct captioning only

### 2. GIT-base (Efficient & Fast - No Prompts)

- **Model** : `microsoft/git-base`
- **Size** : ~350MB
- **VRAM** : ~1-2GB (perfect for GTX 1050)
- **Purpose** : Lightweight and efficient model optimized for speed
- **Speed** : ~1 second/image
- **Prompts** : ❌ No - Direct captioning only
- **Note** : Excellent balance between speed and quality

### 3. Swin-Tiny-DistilGPT2 (Ultra-Lightweight + Prompts) ✨

- **Model** : `yesidcanoc/image-captioning-swin-tiny-distilgpt2`
- **Size** : ~0.15B parameters (~150MB)
- **VRAM** : <1GB (FP16)
- **Purpose** : Ultra-lightweight with Swin Transformer architecture
- **Architecture** : Swin-Tiny encoder + DistilGPT2 decoder
- **Speed** : <1 second/image
- **Prompts** : ✅ Yes - Custom prompts via decoder prefix
- **Note** : Swin architecture provides better hierarchical feature learning than ViT

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
- NVIDIA GPU with CUDA support (tested on GTX 1050)
- **4GB VRAM minimum** (models use quantization)
- ~10GB disk space (for model weights)

## Usage

### 1. Build Docker Image

```bash
make build
```

### 2. Generate Captions with All Models

Run all three VLMs sequentially:

```bash
make run-all
```

This will:

- Generate captions with BLIP-base
- Generate captions with GIT-base
- Generate captions with Swin-Tiny-DistilGPT2
- Save results in separate folders per model
- Total time: <1 minute for 20 images

### 3. Run Individual Models

Run specific VLMs:

```bash
make run-blip            # BLIP-base only (~30 seconds)
make run-git-base        # GIT-base only (~20 seconds)
make run-swin-tiny       # Swin-Tiny only (<20 seconds)
```

### 4. Evaluate Results

Compute metrics for all models:

```bash
make evaluate
```

This will:

- Load generated captions and ground truth
- Compute BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr
- Generate per-category analysis
- Save results to `results/*/metrics.json`

### 5. Generate Comparative Analysis

```bash
make compare
```

Creates cross-model comparison with:

- Side-by-side caption quality
- Metric comparison tables
- Category-specific performance analysis

### 6. Interactive Development

```bash
make shell
```

Opens a bash shell inside the container for interactive development and debugging.

### Additional Commands

- `make gpu-info` - Show GPU information and VRAM usage
- `make clean` - Remove Docker image
- `make clean-results` - Delete all generated results

## Expected Execution Times (GTX 1050)

| Model                | Per Image | Full Dataset (20 images) |
| -------------------- | --------- | ------------------------ |
| BLIP-base            | ~1-2s     | ~30-40s                  |
| GIT-base             | ~1s       | ~20s                     |
| Swin-Tiny-DistilGPT2 | <1s       | <20s                     |
| **Total**            | -         | **<1 minute**            |

## Evaluation Metrics

The following metrics are computed for each model:

- **BLEU-1, BLEU-4** : N-gram precision
- **ROUGE-L** : Longest common subsequence
- **METEOR** : Alignment-based metric
- **CIDEr** : Consensus-based metric for image descriptions

Results include:

- Overall scores across all images
- Per-category breakdowns
- Best/worst performing images per model

## Notes

### VRAM Management

- Models run **sequentially**, not simultaneously
- VRAM is freed between model executions
- Ultra-lightweight models (all <1GB) require no quantization
- ~3GB safety margin maintained on 4GB GPU

### First Run

- Model weights downloaded automatically (~2GB total for all models)
- Download time depends on internet connection (~5-10 minutes first time)
- Weights cached for subsequent runs

### Model Selection Rationale

- **BLIP-base**: Industry-standard baseline, fast, reliable, no prompts
- **GIT-base**: Lightweight and efficient, excellent speed/quality balance, minimal VRAM usage, no prompts
- **Swin-Tiny-DistilGPT2**: Ultra-lightweight with prompt tuning, Swin Transformer for better hierarchical features, trained on COCO

### Quantization Strategy

- **BLIP-base**: FP16 (native)
- **GIT-base**: FP16 (native)
- **Swin-Tiny-DistilGPT2**: FP16 (no quantization needed, <1GB)

### Tested Models (Memory/Loading Issues)

During development, several models were tested but encountered loading or memory issues on GTX 1050 (4GB VRAM):

- **MiniCPM-V 2.0** (`openbmb/MiniCPM-Llama3-V-2_5`): Failed to load within VRAM constraints despite 4-bit quantization. Model hung during initialization after ~1 hour of waiting.

- **Moondream2** (`vikhyatk/moondream2`): Memory initialization issues prevented proper loading. Incompatibility with current transformers version.

- **Qwen2-VL-2B** (`Qwen/Qwen2-VL-2B-Instruct`): Loaded successfully but encountered CUDA Out of Memory error during inference. Model attempted to allocate 4.27GB on a 4GB GPU, failing at the 8th image. While promising, it exceeds the available VRAM.

- **InstructBLIP** (`Salesforce/instructblip-vicuna-7b`): Model download/shard loading extremely slow. After ~1.5 hours, shards failed to load completely. The 7B model size appears too large for efficient loading even with 4-bit quantization on limited hardware.

- **Phi-3-Vision** (`microsoft/Phi-3-vision-128k-instruct`): Shard loading timeout after ~16 minutes. Despite 4-bit quantization reducing the 3.8B model to ~2.5GB, the loading process failed to complete. Model hung during shard loading similar to InstructBLIP.

- **ViT2DistilGPT2** (`sachin/vit2distilgpt2`): Ultra-lightweight VisionEncoderDecoder model that loaded successfully but generated empty or single-character captions. The model appears to not be properly fine-tuned for captioning tasks despite its VisionEncoderDecoder architecture. Removed from final model set.

**Selected Alternatives**: After extensive testing, Swin-Tiny-DistilGPT2 was selected as the prompt-capable model. It's ultra-lightweight (<0.2B parameters), loads instantly, runs in <1GB VRAM, and supports prompt tuning through decoder prefix injection. Combined with BLIP-base and GIT-base, this provides a balanced comparison with two direct captioning models and one prompt-capable model, all working reliably on 4GB VRAM.
