# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset.

## Project Structure

```
practica_3_vlm/
├── Dockerfile              # Docker configuration
├── Makefile               # Build and execution recipes
├── requirements.txt       # Python dependencies
├── src/
│   ├── vlm_inference.py  # VLM caption generation
│   ├── evaluation.py     # Metrics computation
│   └── utils.py          # Helper functions
├── data/
│   └── f1_dataset/       # Dataset from practice 2
└── results/              # Generated captions and metrics
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA support (tested on GTX 1050)
- At least 4GB VRAM (quantized models supported)

## Usage

### 1. Build Docker Image

```bash
make build
```

### 2. Generate Captions

```bash
make run
```

This will:

- Load the VLM model
- Process all images in `data/f1_dataset/images/`
- Generate captions
- Save results to `results/generated_captions.json`

### 3. Evaluate Results

```bash
make evaluate
```

This will:

- Load generated captions and ground truth
- Compute evaluation metrics (BLEU, ROUGE, etc.)
- Save results to `results/metrics_results.json`

### 4. Interactive Development

```bash
make shell
```

Opens a bash shell inside the container for interactive development.

### Additional Commands

- `make gpu-info` - Show GPU information
- `make run-cpu` - Run on CPU (slower, for testing)
- `make clean` - Remove Docker image

## Dataset Format

Expected structure in `data/f1_dataset/`:

```
data/f1_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── captions.json
```

`captions.json` format:

```json
{
  "image_001.jpg": "caption text here",
  "image_002.jpg": "another caption"
}
```

## Notes

- Quantized models are used to fit in low VRAM GPUs
- First run will download model weights (may take time)
- Results are saved in `results/` directory
