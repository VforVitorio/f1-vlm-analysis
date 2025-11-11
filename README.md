# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset comparing four different architectures.

## Models Used

This project implements and compares **four VLM architectures** optimized for 4GB VRAM:

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

### 3. InstructBLIP (Prompt Tuning Capable) 🆕

- **Model** : `Salesforce/instructblip-vicuna-7b`
- **Size** : ~7B parameters (4-bit quantized to ~3.5GB)
- **VRAM** : ~3-3.5GB with 4-bit quantization
- **Purpose** : Instruction-based captioning with customizable prompts
- **Speed** : ~2-3 seconds/image
- **Prompts** : ✅ Yes - Accepts custom instruction prompts
- **Note** : Perfect for prompt engineering experiments

### 4. Phi-3-Vision (Efficient Prompt Tuning) 🆕

- **Model** : `microsoft/Phi-3-vision-128k-instruct`
- **Size** : ~3.8B parameters (4-bit quantized to ~2.5GB)
- **VRAM** : ~2.5-3GB with 4-bit quantization
- **Purpose** : Microsoft's efficient multimodal model with strong instruction-following
- **Speed** : ~1-2 seconds/image
- **Prompts** : ✅ Yes - Highly flexible conversational prompts
- **Note** : Best efficiency for prompt-based captioning

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
│   │   ├── instructblip_model.py  # InstructBLIP captioner class
│   │   └── phi3_vision_model.py   # Phi-3-Vision captioner class
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
    ├── instructblip/        # InstructBLIP results
    │   ├── generated_captions.json
    │   └── metrics.json
    ├── phi3-vision/         # Phi-3-Vision results
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

Run all four VLMs sequentially:

```bash
make run-all
```

This will:

- Generate captions with BLIP-base
- Generate captions with GIT-base
- Generate captions with InstructBLIP
- Generate captions with Phi-3-Vision
- Save results in separate folders per model
- Total time: ~2-3 minutes for 20 images

### 3. Run Individual Models

Run specific VLMs:

```bash
make run-blip           # BLIP-base only (~30 seconds)
make run-git-base       # GIT-base only (~20 seconds)
make run-instructblip   # InstructBLIP only (~1 minute)
make run-phi3-vision    # Phi-3-Vision only (~40 seconds)
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

| Model        | Per Image | Full Dataset (20 images) |
| ------------ | --------- | ------------------------ |
| BLIP-base    | ~1-2s     | ~30-40s                  |
| GIT-base     | ~1s       | ~20s                     |
| InstructBLIP | ~2-3s     | ~40-60s                  |
| Phi-3-Vision | ~1-2s     | ~30-40s                  |
| **Total**    | -         | **~2-3 minutes**         |

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
- 4-bit quantization used for InstructBLIP and Phi-3-Vision
- ~500MB safety margin maintained

### First Run

- Model weights downloaded automatically (~8-10GB total for all models)
- Download time depends on internet connection (~10-20 minutes first time)
- Weights cached for subsequent runs

### Model Selection Rationale

- **BLIP-base**: Industry-standard baseline, fast, reliable, no prompts
- **GIT-base**: Lightweight and efficient, excellent speed/quality balance, minimal VRAM usage, no prompts
- **InstructBLIP**: Prompt tuning capable, instruction-based captioning, good for experiments
- **Phi-3-Vision**: Efficient prompt tuning, Microsoft's latest multimodal model, best efficiency/capability ratio

### Quantization Strategy

- **BLIP-base**: FP16 (native)
- **GIT-base**: FP16 (native, CPU fallback)
- **InstructBLIP**: 4-bit NF4 (aggressive quantization to fit 7B model)
- **Phi-3-Vision**: 4-bit NF4 (optimal balance for 3.8B model)

### Tested Models (Memory Issues)

During development, several models were tested but encountered loading or memory issues on GTX 1050 (4GB VRAM):

- **MiniCPM-V 2.0** (`openbmb/MiniCPM-Llama3-V-2_5`): Failed to load within VRAM constraints despite 4-bit quantization. Model hung during initialization after ~1 hour of waiting.

- **Moondream2** (`vikhyatk/moondream2`): Memory initialization issues prevented proper loading. Incompatibility with current transformers version.

- **Qwen2-VL-2B** (`Qwen/Qwen2-VL-2B-Instruct`): Loaded successfully but encountered CUDA Out of Memory error during inference. Model attempted to allocate 4.27GB on a 4GB GPU, failing at the 8th image. While promising, it exceeds the available VRAM.

**Selected Alternatives**: InstructBLIP and Phi-3-Vision were chosen as prompt-capable alternatives that successfully fit within 4GB VRAM constraints while offering instruction-following capabilities.
