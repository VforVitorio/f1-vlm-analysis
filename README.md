# Practice 3: VLM Image Captioning

Vision-Language Model (VLM) based image captioning system for Formula 1 dataset comparing three ultra-lightweight architectures.

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

---

## Models Used

This project implements and compares **three first-generation VLM architectures** optimized for 4GB VRAM (GTX 1050):

### 1. BLIP-base (Baseline - No Prompts)

- **Model** : `Salesforce/blip-image-captioning-base`
- **Architecture**: ViT encoder + BERT decoder (VisionEncoderDecoder)
- **Size** : ~450MB (FP16)
- **VRAM** : ~1.5GB
- **Purpose** : Establishes baseline performance with proven architecture
- **Speed** : ~1-2 seconds/image
- **Prompts** : ❌ No - Direct captioning only
- **Generation**: COCO-trained, deterministic beam search

### 2. GIT-base (Efficient & Fast - No Prompts)

- **Model** : `microsoft/git-base`
- **Architecture**: ViT encoder + GPT-2 decoder (VisionEncoderDecoder)
- **Size** : ~350MB
- **VRAM** : ~1-2GB (perfect for GTX 1050)
- **Purpose** : Lightweight and efficient model optimized for speed
- **Speed** : ~1 second/image
- **Prompts** : ❌ No - Direct captioning only
- **Note** : Excellent balance between speed and quality

### 3. Swin-Tiny-DistilGPT2 (Ultra-Lightweight) ⚠️

- **Model** : `yesidcanoc/image-captioning-swin-tiny-distilgpt2`
- **Architecture** : Swin-Tiny encoder + DistilGPT2 decoder (VisionEncoderDecoder)
- **Size** : ~0.15B parameters (~150MB)
- **VRAM** : <1GB (FP16)
- **Purpose** : Experimental model to test Swin Transformer architecture vs ViT
- **Speed** : <1 second/image
- **Prompts** : ❌ No - Direct captioning only
- **Result** : ⚠️ **Poor quality on F1 dataset** - Generates generic COCO captions unrelated to racing context
- **Note** : Included as negative control experiment. Despite same VisionEncoderDecoder architecture as BLIP/GIT, this specific checkpoint produces irrelevant captions (baseball players, tennis racquets, motorcycles instead of F1 cars). Demonstrates that **model checkpoint quality matters more than architecture** - not all VisionEncoderDecoder models trained on COCO generalize equally well to domain-specific datasets.
- **Prompt Tuning Attempt**: Extensively tested category-specific prompts via `decoder_input_ids` prefix injection (both long instructions and short semantic primes). All approaches caused severe output degradation - model interprets prompts as literal caption beginnings, generating nonsensical text like "emotioning MOTORCYCLE PLAYING TENNIS" or "stop scene with orange dolls". VisionEncoderDecoder architecture technically supports decoder prefixes, but this specific checkpoint was not trained for it. Reverted to direct generation for coherent output.

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

| Model                | Per Image | Full Dataset (20 images) | Quality |
| -------------------- | --------- | ------------------------ | ------- |
| BLIP-base            | ~1-2s     | ~30-40s                  | ✅ Good |
| GIT-base             | ~1s       | ~20s                     | ✅ Good |
| Swin-Tiny-DistilGPT2 | <1s       | <20s                     | ❌ Poor |
| **Total**            | -         | **<1 minute**            | -       |

**Note**: Swin-Tiny executes successfully but produces low-quality captions irrelevant to F1 context (e.g., "baseball player", "tennis racquet"). Kept as experimental baseline to demonstrate checkpoint quality importance.

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

### Architecture Choice: Why First-Gen VLMs?

This project uses **VisionEncoderDecoder models** (BLIP, GIT, Swin-Tiny) instead of modern LLM-based VLMs due to **VRAM constraints on GTX 1050 (4GB)**:

**First-Gen VLMs (Used)**:

- ✅ VRAM: <2GB per model
- ✅ Load time: Instant (<10 seconds)
- ✅ Inference: Very fast (<2s per image)
- ❌ Limitation: Cannot follow instructions, no prompt tuning support
- ❌ Limitation: Decoder prefix prompting degrades output quality

**Modern LLM-based VLMs (Failed)**:

- ❌ VRAM: >4GB even with 4-bit quantization
- ❌ Load time: Extremely slow (timeouts after 16min-1.5hr)
- ❌ Inference: OOM errors during generation
- ✅ Advantage: Would support full instruction following (if hardware allowed)

**Trade-off**: We sacrifice instruction-following and prompt tuning capabilities for reliability and speed on limited hardware. First-gen VLMs produce quality captions through direct image-to-text generation but cannot be guided with prompts or instructions.

### Why Not Smaller Modern LLM-based VLMs?

While there exist newer ultra-compact LLM-based VLMs (2-3B parameters) like **PaliGemma-3B**, **SmolVLM-Instruct (2B)**, **MobileVLM V2 (1.7B)**, or **LLaVA-Phi-3-mini**, they are **not viable for 4GB VRAM** due to fundamental resource constraints:

**Physical VRAM Requirements (minimum for modern LLM-VLM)**:

- Base LLM (1-2B params): ~2GB in 4-bit quantization
- Vision Encoder (400-500M): ~400-500MB
- Activation memory during inference: ~1-1.5GB overhead
- **Total minimum**: ~3.5-4GB → **No safety margin on 4GB GPU**

**Why They Fail in Practice**:

- Qwen2-VL-2B tested: **OOM at 4.27GB** during 8th image
- Phi-3-Vision-3.8B tested: **Shard loading timeout** after 16 minutes
- Models with 2-3GB theoretical footprint **exceed 4GB during generation**
- Quantization overhead + batch processing = instant crash
- Extremely slow loading times (shards, quantization) = poor UX

**First-Gen VLMs Comparison**:

- BLIP/GIT/Swin-Tiny: <2GB per model, ~3GB free margin
- Instant loading, stable inference, production-proven
- Trade instruction-following for **100% reliability** on limited hardware

**Conclusion**: For 4GB VRAM, first-gen VisionEncoderDecoder models are the **only practical choice**. Modern LLM-based VLMs require minimum 6-8GB VRAM for stable operation.

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

- **BLIP-base**: Industry-standard first-gen VLM, VisionEncoderDecoder with ViT+BERT, proven reliability on diverse datasets
- **GIT-base**: Lightweight VisionEncoderDecoder with ViT+GPT2, excellent speed/quality balance, strong generalization
- **Swin-Tiny-DistilGPT2**: Experimental VisionEncoderDecoder with Swin+DistilGPT2 - **negative control showing checkpoint quality matters more than architecture**

All three models share the same **VisionEncoderDecoder architecture** (vision encoder → cross-attention → language decoder) which enables reliable operation on 4GB VRAM. None support prompt tuning - all use direct image-to-text generation trained on COCO captions.

**Key Finding**: Despite identical architecture, **BLIP and GIT produce relevant F1 captions** while **Swin-Tiny generates generic COCO descriptions** (baseball, tennis, motorcycles). This demonstrates that model checkpoint training quality and generalization capability are more critical than architectural choices when working with domain-specific datasets. Swin-Tiny is kept in the comparison as experimental evidence of this phenomenon.

### Quantization Strategy

- **BLIP-base**: FP16 (native)
- **GIT-base**: FP16 (native)
- **Swin-Tiny-DistilGPT2**: FP16 (no quantization needed, <1GB)

### Tested Modern LLM-based VLMs (Failed on 4GB VRAM)

During development, several **modern LLM-based multimodal models** were tested but all failed due to hardware limitations on GTX 1050 (4GB VRAM). These represent the newer generation of VLMs with full instruction-following capabilities:

- **Qwen2-VL-2B** (`Qwen/Qwen2-VL-2B-Instruct`) [LLM-based]: Loaded successfully but encountered **CUDA Out of Memory** error during inference. Model attempted to allocate 4.27GB on a 4GB GPU, failing at the 8th image. Architecture: Vision encoder → Qwen2-2B LLM. Would support full natural language instructions if hardware allowed.

- **Phi-3-Vision** (`microsoft/Phi-3-vision-128k-instruct`) [LLM-based]: **Shard loading timeout** after ~16 minutes. Despite 4-bit quantization reducing the 3.8B Phi-3 LLM to ~2.5GB, the loading process failed to complete. Architecture: Vision encoder → Phi-3-mini LLM.

- **InstructBLIP** (`Salesforce/instructblip-vicuna-7b`) [LLM-based]: Model download/shard loading extremely slow. After **~1.5 hours**, shards failed to load completely. Architecture: Q-Former → Vicuna-7B LLM. The 7B LLM size is too large for efficient loading even with 4-bit quantization.

- **MiniCPM-V 2.0** (`openbmb/MiniCPM-Llama3-V-2_5`) [LLM-based]: Failed to load within VRAM constraints despite 4-bit quantization. Model **hung during initialization after ~1 hour** of waiting. Architecture: Vision encoder → Llama3 LLM.

- **Moondream2** (`vikhyatk/moondream2`) [LLM-based]: Memory initialization issues prevented proper loading. Incompatibility with current transformers version. Small LLM-based model but implementation issues.

### Tested First-Gen VLMs (Also Failed)

- **ViT2DistilGPT2** (`sachin/vit2distilgpt2`) [VisionEncoderDecoder]: Ultra-lightweight model that loaded successfully but generated **empty or single-character captions**. Despite being the same VisionEncoderDecoder architecture as working models, this checkpoint appears to not be properly fine-tuned for captioning tasks. Removed from final model set.

**Key Takeaway**: All modern LLM-based VLMs (Qwen, Phi, InstructBLIP, MiniCPM) **require >4GB VRAM** even with aggressive quantization. First-gen VisionEncoderDecoder models (BLIP, GIT, Swin-Tiny) are the only viable option for GTX 1050, trading instruction-following capabilities for reliability and speed.
