# Autoresearch Experiment Log

## Overview
Autonomous LLM pretraining research using [autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) fork (Windows + consumer NVIDIA GPUs).

## System
| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM, Ada Lovelace (SM 8.9) |
| CPU | AMD Ryzen ~4.3 GHz |
| RAM | 32 GB |
| OS | Windows 11 Pro |
| CUDA | 13.2 |
| Python | 3.13 (managed by uv 0.10.9) |
| PyTorch | 2.9.1 (CUDA 12.8) |

## Runtime Configuration (auto-detected)
| Parameter | Value |
|---|---|
| GPU Profile | ada-10-15gb |
| AMP dtype | bfloat16 |
| Attention | PyTorch SDPA |
| torch.compile | disabled (eager mode) |
| Activation checkpointing | enabled |
| Train batch size | 8 (autotuned) |
| Eval batch size | 8 |
| Dataset | TinyStories GPT-4 clean |
| Vocab size | 8,192 |
| Time budget | 300s (5 min per experiment) |

## Model Architecture (default)
| Parameter | Value |
|---|---|
| Depth (layers) | 8 |
| Embedding dim | 512 |
| Heads | 4 |
| KV Heads | 4 |
| Window pattern | SSSL |
| Total params | 50.3M |

## Setup Steps (reproducibility)
```powershell
# 1. Clone the repo
git clone https://github.com/jsegov/autoresearch-win-rtx.git
cd autoresearch-win-rtx

# 2. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install dependencies
uv sync

# 4. Prepare data (download TinyStories + train BPE tokenizer)
uv run prepare.py

# 5. Smoke test
uv run train.py --smoke-test

# 6. Full baseline run (5 min)
uv run train.py
```

## Experiment Log
See [experiments.md](experiments.md) for the full experiment history.
