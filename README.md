# Aare HITL

A command-line tool for fine-tuning language models with human-in-the-loop (HITL) validation.

## Overview

Aare makes it easy to:

- Fine-tune open-source LLMs on your domain-specific data
- Compare base model vs fine-tuned outputs interactively
- Validate training datasets before use
- Use GPU acceleration (MPS on Mac, CUDA on Linux/Windows)

## Installation

```bash
git clone https://github.com/aare-ai/aare-hitl.git
cd aare-hitl
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Create a JSON file with instruction/output pairs:

```json
[
  {
    "instruction": "My cake didn't rise. What went wrong?",
    "output": "Common causes: expired baking powder, overmixing the batter, oven temperature too low, or opening the oven door too early."
  }
]
```

Validate your dataset:

```bash
aare data validate data/my_dataset.json
```

### 2. Train a Model

Create a training config (`train.yaml`):

```yaml
# Training config
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset: ./data/my_dataset.json
output_dir: ./output

# Training parameters
method: lora
epochs: 3
batch_size: 2
learning_rate: 0.0002
lora_rank: 16

# Device: auto | cpu | mps | cuda
device: auto
```

Run training:

```bash
aare train --config train.yaml
```

### 3. Compare Base vs Fine-tuned

Create a compare config (`compare.yaml`):

```yaml
# Compare config
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
adapter: ./output
dataset: ./data/my_dataset.json
output_dir: ./compare_results

# Device: auto | cpu | mps | cuda
device: auto
```

Run interactive HITL comparison:

```bash
aare compare --config compare.yaml
```

This launches an interactive session where you can:

- View each training sample
- Press `c` to generate and compare base vs fine-tuned outputs
- Accept (`a`), reject (`r`), or skip (`s`) samples
- Navigate with `n`/`p` for next/previous
- Quit with `q`

Accepted and rejected samples are saved to `compare_results/accepted.json` and `compare_results/rejected.json`.

### 4. Generate Text

Create a generate config (`generate.yaml`):

```yaml
# Generate config
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
adapter: ./output
prompt: "My cake didn't rise. What went wrong?"
max_tokens: 512

# Device: auto | cpu | mps | cuda
device: auto
```

Generate:

```bash
aare generate --config generate.yaml
```

## Commands

| Command | Description |
|---------|-------------|
| `aare train --config <file>` | Train a model using LoRA |
| `aare compare --config <file>` | Interactive HITL comparison |
| `aare generate --config <file>` | Generate from a trained model |
| `aare data validate <file>` | Validate a dataset file |
| `aare data import <csv> -o <json>` | Import CSV (e.g., from Google Sheets) |
| `aare data merge <base> --remove <file> -o <output>` | Remove rejected samples |
| `aare data merge <base> --add <file> -o <output>` | Add new samples |

## Global Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug logging |
| `--log-format text\|json` | Output format (default: text) |
| `--version` | Show version |

## Config Format

Configs must be **YAML** (`.yaml` or `.yml`). YAML supports comments and is easier to read.

```yaml
# Training config with comments
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset: ./data/my_dataset.json
```

Datasets remain JSON format for compatibility with standard tools.

## Config Reference

### Training Config

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | - | HuggingFace model ID |
| `dataset` | Yes | - | Path to JSON dataset |
| `output_dir` | No | `./output` | Where to save the adapter |
| `method` | No | `lora` | `lora`, `qlora`, or `full` |
| `epochs` | No | `1` | Number of training epochs |
| `batch_size` | No | `2` | Training batch size |
| `learning_rate` | No | `0.0002` | Learning rate |
| `lora_rank` | No | `8` | LoRA rank (higher = more capacity) |
| `device` | No | `auto` | `auto`, `cpu`, `mps`, or `cuda` |

### Compare Config

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | - | HuggingFace model ID |
| `dataset` | Yes | - | Path to JSON dataset |
| `adapter` | No | `./output` | Path to LoRA adapter |
| `output_dir` | No | `./compare_results` | Where to save results |
| `device` | No | `auto` | `auto`, `cpu`, `mps`, or `cuda` |

### Generate Config

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | - | HuggingFace model ID |
| `prompt` | Yes | - | Input prompt |
| `adapter` | No | `""` | Path to LoRA adapter |
| `max_tokens` | No | `512` | Maximum tokens to generate (up to 1024+) |
| `device` | No | `auto` | `auto`, `cpu`, `mps`, or `cuda` |

## Device Selection

Aare automatically detects and uses the best available device:

| Device | Platform | Notes |
|--------|----------|-------|
| `auto` | Any | Auto-detect best device (default) |
| `mps` | macOS | Apple Silicon GPU (M1/M2/M3/M4) |
| `cuda` | Linux/Windows | NVIDIA GPU |
| `cpu` | Any | CPU only (slowest) |

GPU acceleration uses float16 precision for faster inference. CPU uses float32.

Example for Mac with Apple Silicon:

```yaml
device: mps  # Use Metal Performance Shaders
```

Example for NVIDIA GPU:

```yaml
device: cuda  # Use CUDA
```

## Dataset Format

Aare uses the standard Alpaca/instruction format (JSON):

```json
[
  {"instruction": "...", "output": "..."},
  {"instruction": "...", "output": "..."}
]
```

Also accepts:

- `prompt` instead of `instruction`
- `response` instead of `output`
- `text` for raw text samples

### Using Google Sheets

You can manage training data in Google Sheets and import it:

1. Create a Google Sheet with columns `instruction` and `output`
2. Export: **File → Download → Comma-separated values (.csv)**
3. Import to JSON:

```bash
aare data import mydata.csv -o dataset.json
```

Custom column names:

```bash
aare data import mydata.csv -o dataset.json --instruction-col question --output-col answer
```

## Supported Models

Any HuggingFace causal LM works. Tested models:

| Model | Size | Notes |
|-------|------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Fast, good for testing |
| `Qwen/Qwen2-0.5B` | 0.5B | Very small, open |
| `Qwen/Qwen2-1.5B` | 1.5B | Small, open |
| `microsoft/phi-2` | 2.7B | Small but capable |
| `mistralai/Mistral-7B-v0.3` | 7B | Good quality |
| `meta-llama/Llama-3.1-8B` | 8B | Requires HF login |

## Training Methods

| Method | Memory | Speed | Use Case |
|--------|--------|-------|----------|
| `lora` | ~4GB | Fast | Default choice |
| `qlora` | ~2GB | Medium | Limited memory |
| `full` | ~16GB+ | Slow | Best quality |

## How It Works

### Training

1. **Load model** from HuggingFace
2. **Apply LoRA** - add small trainable adapter layers
3. **Tokenize dataset** - convert text to model input format
4. **Train** - update only adapter weights
5. **Save adapter** - ~10-50MB vs 2-8GB for full model

### LoRA (Low-Rank Adaptation)

Instead of updating all model parameters (billions), LoRA injects small trainable matrices into specific layers. This reduces memory from 16GB+ to ~4GB while maintaining quality.

### HITL Comparison

The compare command helps you validate training effectiveness:

1. Show training sample prompt
2. Generate responses from base model and fine-tuned model
3. You decide if the fine-tuned output is better
4. Track accept/reject statistics
5. Save results to JSON files for further training iterations

### HITL Feedback Loop

The full iterative workflow:

```
┌─────────────────────────────────────────────────────────┐
│  1. Train                                               │
│     aare train --config train.yaml                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. Compare (HITL)                                      │
│     aare compare --config compare.yaml                  │
│     → Saves accepted.json and rejected.json             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. Curate Dataset                                      │
│     aare data merge original.json \                     │
│       --remove rejected.json \                          │
│       -o cleaned.json                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Retrain with curated data                           │
│     (Update train.yaml to use cleaned.json)             │
│     aare train --config train.yaml                      │
└─────────────────────────────────────────────────────────┘
                          ↓
                    (Repeat until satisfied)
```

**Merge options:**

```bash
# Remove rejected samples
aare data merge data.json --remove rejected.json -o cleaned.json

# Add new samples
aare data merge data.json --add new_samples.json -o expanded.json

# Both at once
aare data merge data.json --remove rejected.json --add accepted.json -o curated.json
```

## Dependencies

Core:

- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- PyYAML

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AARE_LOG_LEVEL` | Log level | `INFO` |
| `AARE_LOG_FORMAT` | `text` or `json` | `text` |
| `AARE_DATA_DIR` | Data directory | `./data` |
| `AARE_MODELS_DIR` | Models directory | `./models` |

## License

MIT
