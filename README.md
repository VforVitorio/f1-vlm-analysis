# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset comparing three different architectures.

## Models Used

This project implements and compares **three VLM architectures** optimized for 4GB VRAM:

### 1. BLIP-base (Baseline)

- **Model** : `Salesforce/blip-image-captioning-base`
- **Size** : ~450MB (FP16)
- **VRAM** : ~1.5GB
- **Purpose** : Establishes baseline performance with proven architecture
- **Speed** : ~1-2 seconds/image

### 2. Moondream2 (Efficient)

- **Model** : `vikhyatk/moondream2`
- **Size** : ~1.6B parameters
- **VRAM** : ~1.5-2GB
- **Purpose** : Modern efficient VLM optimized for low-resource environments
- **Speed** : ~2-3 seconds/image

### 3. MiniCPM-V 2.0 (Advanced)

- **Model** : `openbmb/MiniCPM-V-2`
- **Size** : ~2.4B parameters (4-bit quantized)
- **VRAM** : ~3GB
- **Purpose** : State-of-the-art compact VLM with strong detail recognition
- **Speed** : ~4-6 seconds/image
- **Note** : Uses aggressive 4-bit quantization to fit in 4GB VRAM

## Project Structure

```
practica_3_vlm/
├── Dockerfile                # Docker configuration with CUDA support
├── Makefile                  # Build and execution recipes
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── dataset_loader.py    # Load F1 dataset with categories
│   ├── vlm_inference.py     # VLM caption generation (3 models)
│   ├── evaluation.py        # Metrics computation (BLEU, ROUGE, etc.)
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
    ├── moondream/           # Moondream2 results
    │   ├── generated_captions.json
    │   └── metrics.json
    ├── minicpm/             # MiniCPM-V results
    │   ├── generated_captions.json
    │   └── metrics.json
    └── comparison/          # Cross-model comparison
        └── comparative_analysis.json
```

## Dataset Structure

> [!WARNING] > **Dataset Not Included**
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
- Generate captions with Moondream2
- Generate captions with MiniCPM-V 2.0
- Save results in separate folders per model
- Total time: ~3-5 minutes for 20 images

### 3. Run Individual Models

Run specific VLMs:

```bash
make run-blip        # BLIP-base only (~30 seconds)
make run-moondream   # Moondream2 only (~1 minute)
make run-minicpm     # MiniCPM-V only (~2 minutes)
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

| Model         | Per Image | Full Dataset (20 images) |
| ------------- | --------- | ------------------------ |
| BLIP-base     | ~1-2s     | ~30-40s                  |
| Moondream2    | ~2-3s     | ~40-60s                  |
| MiniCPM-V 2.0 | ~4-6s     | ~80-120s                 |
| **Total**     | -         | **~3-4 minutes**         |

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

- Models run **sequentially** , not simultaneously
- VRAM is freed between model executions
- 4-bit quantization used for MiniCPM-V to fit in 4GB
- ~500MB safety margin maintained

### First Run

- Model weights downloaded automatically (~3-4GB total)
- Download time depends on internet connection (~5-10 minutes)
- Weights cached for subsequent runs

### Model Selection Rationale

- **BLIP** : Industry-standard baseline, fast, reliable
- **Moondream2** : Efficient modern architecture, good speed/quality balance
- **MiniCPM-V** : Best quality within VRAM constraints, detailed descriptions

### Quantization Strategy

- BLIP-base: FP16 (native)
- Moondream2: FP16 (native)
- MiniCPM-V: 4-bit (aggressive, necessary for 4GB limit)
