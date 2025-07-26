# 🏷️ Gemma + QLoRA Multi‑Label StackExchange Classifier

Fine‑tuned **Gemma 2B Instruction** with **QLoRA** and 4‑bit quantization to automatically tag StackExchange questions with all relevant topics.  
The model delivers high accuracy while cutting GPU memory usage in half, making large‑model multi‑label classification feasible on a single consumer GPU.

---

## 📈 Key Results

| Metric (hold‑out) | Score |
|-------------------|-------|
| **Macro F1**      | **0.82** |
| Precision         | 0.86 |
| Recall            | 0.80 |
| Parameters        | 2.2 B (base) / **19 M trainable** |
| VRAM Footprint    | **~7 GB** (4‑bit QLoRA) |

---

## 🔧 Technologies Used

- **Gemma 2B‑Instruct** – Google’s open instruction‑tuned LLM  
- **QLoRA** – Parameter‑efficient 4‑bit Low‑Rank Adaptation  
- **PEFT** – Hugging Face’s PEFT library for lightweight fine‑tuning  
- **bitsandbytes** – 4‑bit & 8‑bit optimizers / quantization  
- **PyTorch 2.1** + **🤗 Transformers v4.54**  
- **Datasets** – Hugging Face Datasets for streaming StackExchange data  
- **Weights & Biases** – experiment tracking (optional)

---

## 🗃️ Dataset

- **Pre‑processing**:  
  1. Filtered English‑language questions with ≥1 accepted answer  
  2. Kept questions with ≤6 tags (multi‑label)  
  3. Tokenized titles + bodies using `AutoTokenizer.from_pretrained("google/gemma-2b-it")`  
- **Train / Val / Test**: 80 % / 10 % / 10 %

---

## 🚀 Quick Start

```bash
pip install "transformers>=4.54" "accelerate>=0.25" datasets peft bitsandbytes einops
