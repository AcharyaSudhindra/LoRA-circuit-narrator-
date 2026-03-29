# LoRA Circuit Narrator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**A fine-tuned language model that explains digital circuits and VLSI concepts through creative, intuitive stories.**

*"Instead of dry definitions, every concept becomes an analogy you actually remember."*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Dataset](#-dataset) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## Overview

**LoRA Circuit Narrator llm** is a domain-specific language model fine-tuned on `EleutherAI/gpt-neo-125M` using **Low-Rank Adaptation (LoRA)** to explain digital electronics and VLSI design concepts through creative analogies and storytelling.

The model was trained on a hand-crafted dataset of around **300 prompt-response pairs** covering topics from basic logic gates , memory systems, CPU microarchitecture, embedded systems, semiconductor physics, and high-speed communication protocols.

The goal is to make complex engineering concepts accessible to students and hobbyists by presenting them as memorable real-world analogies rather than abstract technical definitions.

---

## Features

-  **Domain-specific fine-tuning** on 311 VLSI and digital electronics concepts
-  **LoRA fine-tuning** — trains only 0.24% of model parameters, making it fast and memory-efficient
-  **Interactive Gradio web UI** with adjustable generation parameters
-  **GPU accelerated training** with fp16 mixed precision support
-  **Lightweight base model** — GPT-Neo 125M, runs on any consumer hardware
-  **Reproducible pipeline** — dataset → train → inference in three commands

---

## Demo

Type a prompt in the format:

```
Explain a <concept> in a story
```

**Example input:**
```
Explain a pipeline in a story
```

**Example output:**
```
Imagine a car wash with five stages — rinse, soap, scrub, rinse again, and dry.
Instead of waiting for one car to complete all five stages before the next car enters,
a new car enters stage one as soon as the previous car moves to stage two.
Five cars are being processed simultaneously. A CPU pipeline works the same way —
different instructions are at different stages of execution at the same time,
dramatically increasing throughput.
```

---

##  Project Structure

```
circuit-story-llm/
│
├── dataset/
│   └── circuit_story.jsonl       # 311 prompt-response training pairs
│
├── train.py                      # Fine-tuning script (LoRA + GPT-Neo-125M)
├── app.py                        # Gradio web UI for inference
├── inference.py                  # Command-line inference script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes model weights and cache
└── README.md                     # This file
```

> **Note:** The `results/` directory containing trained model weights is excluded from version control due to size (~500MB). Run `train.py` to generate it locally.

---

##  Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended) or CPU
- 4GB+ RAM minimum, 8GB+ recommended

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/circuit-story-llm.git
cd circuit-story-llm
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) GPU support for RTX 40/50 series

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Usage

### Train the model

```bash
python train.py
```

Expected output:
```
GPU detected: NVIDIA GeForce RTX 5060
Dataset loaded: 311 examples
trainable params: 589,824 || all params: 125,488,128 || trainable%: 0.47
 Starting training on CUDA...
{'loss': 2.14, 'epoch': 1.0}
{'loss': 0.89, 'epoch': 10.0}
{'loss': 0.21, 'epoch': 30.0}
{'loss': 0.06, 'epoch': 50.0}
Training complete. Model saved to ./results
```

Training time: ~4 minutes on RTX 5060 | ~45 minutes on CPU

### Launch the web UI

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`

### Command-line inference

```bash
python inference.py
```

---

## Dataset

The dataset consists of **311 hand-crafted prompt-response pairs** covering the following topic areas:

| Category | Topics Covered | Count |
|---|---|---|
| **Digital Logic** | Logic gates, Boolean algebra, Karnaugh maps, flip-flops, latches, counters | 42 |
| **Computer Architecture** | ALU, pipeline, cache, branch prediction, out-of-order execution, registers | 38 |
| **Memory Systems** | SRAM, DRAM, ROM, Flash, cache hierarchy, virtual memory, TLB | 35 |
| **VLSI Design Flow** | RTL, synthesis, place & route, DRC, LVS, sign-off, floorplanning | 40 |
| **Semiconductor Physics** | MOSFET, CMOS, p-n junction, doping, photolithography, electromigration | 28 |
| **Timing & Clocking** | Setup/hold time, clock skew, metastability, PLL, clock tree | 30 |
| **Embedded Systems** | GPIO, PWM, UART, SPI, I2C, DMA, watchdog, RTOS | 35 |
| **Signal Integrity** | Crosstalk, impedance matching, differential signaling, jitter, eye diagram | 28 |
| **Protocols & Interconnects** | PCIe, USB, Ethernet, I2C, SPI, UART | 20 |
| **Advanced Topics** | Chiplets, heterogeneous integration, Moore's law, dark silicon, FinFET | 15 |

### Dataset Format

Each entry is a JSON object on a single line:

```json
{
  "prompt": "Explain a pipeline in a story",
  "response": "Imagine a car wash with five stages..."
}
```

---

## Architecture

### Base Model

| Property | Value |
|---|---|
| Model | `EleutherAI/gpt-neo-125M` |
| Parameters | 125 million |
| Architecture | GPT-Neo (causal language model) |
| Context length | 2048 tokens |

### Fine-tuning Method — LoRA

Low-Rank Adaptation freezes the original model weights and injects trainable rank-decomposition matrices into the attention layers. This reduces trainable parameters from 125M to just ~590K (0.47%).

| LoRA Hyperparameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha (lora_alpha) | 32 |
| Target modules | `q_proj`, `v_proj` |
| Dropout | 0.05 |
| Task type | `CAUSAL_LM` |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 4 |
| Learning rate | 5e-4 |
| Optimizer | AdamW |
| Precision | fp16 (GPU) / fp32 (CPU) |
| Sequence length | 256 tokens |
| Framework | HuggingFace Transformers + PEFT |

---

## 📈 Results

| Metric | Value |
|---|---|
| Final training loss | **0.06** |
| Training time (RTX 5060) | ~4 minutes |
| Trainable parameters | 589,824 (0.47%) |
| Model size (LoRA weights only) | ~4.5 MB |
| Full merged model size | ~490 MB |

The model successfully learned the story-based explanation style and generates coherent, factually grounded analogies for all 311 trained concepts.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base LLM | GPT-Neo 125M (EleutherAI) |
| Fine-tuning | PEFT / LoRA |
| Training framework | HuggingFace Transformers |
| Dataset loading | HuggingFace Datasets |
| Web UI | Gradio |
| Deep learning | PyTorch |
| GPU acceleration | CUDA 12.8 |
| Language | Python 3.10 |

---

## Requirements

```
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
accelerate>=0.29.0
gradio>=4.0.0
torch>=2.0.0
```

---

## Roadmap

- [ ] Push trained weights to Hugging Face Hub
- [ ] Host live demo on Hugging Face Spaces
- [ ] Expand dataset to 500+ entries
- [ ] Fine-tune a larger base model (GPT-Neo 1.3B)
- [ ] Add support for Verilog/VHDL code generation
- [ ] Implement RAG for real-time datasheet lookup
- [ ] Build a VS Code extension for inline VLSI explanations

---

## Contributing

Contributions are welcome! If you want to add more circuit story entries to the dataset, follow the format in `dataset/circuit_story.jsonl` and open a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Sudhindra**
VLSI Enthusiast 

---

<div align="center">

*Built as a side project exploring the intersection of NLP and VLSI education.*

⭐ If you found this useful, consider starring the repository!

</div>
