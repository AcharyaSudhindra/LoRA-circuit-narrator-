# LoRA Circuit Narrator

**LoRA Circuit Narrator** is a technical project that applies **parameter‑efficient fine‑tuning (LoRA)** to the GPT‑Neo 125M language model, adapting it for the specialized domain of **digital logic and VLSI system design**. The system generates structured, context‑aware explanations of registers, flip‑flops, counters, multiplexers, and arithmetic logic units, presented through a narrative interface for accessibility and demonstration.

---

##  Technical Objectives
- Implement LoRA fine‑tuning for efficient adaptation of GPT‑Neo to a circuit‑focused dataset.
- Construct a domain‑specific dataset of digital logic concepts in Q/A format.
- Validate generated outputs against standard VLSI definitions and design practices.
- Provide an interactive inference interface for demonstration and reproducibility.
- Document the workflow for extension to larger models and datasets.

---

##  Architecture
- **Base Model**: GPT‑Neo 125M (EleutherAI).  
- **Fine‑Tuning Method**: LoRA (Low‑Rank Adaptation).  
- **Frameworks**: Hugging Face Transformers, PEFT, PyTorch.  
- **Dataset**: JSONL file containing structured Q/A pairs on digital circuits.  
- **Deployment**: Gradio web interface with adjustable inference parameters.  

---
