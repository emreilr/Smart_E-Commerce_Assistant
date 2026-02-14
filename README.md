# 🛍️ Smart E-Commerce Assistant (LLM-Powered)

**A specialized AI system for analyzing Turkish e-commerce customer reviews, powered by fine-tuned Large Language Models (LLM).**

This project demonstrates the fine-tuning of the **Trendyol-LLM-7b-base** model using **LoRA (Low-Rank Adaptation)** to accurately detect **Sentiment** (Positive/Negative/Neutral) and **User Intent** (Returns, Complaints, Inquiries, Technical Issues, etc.) from Turkish marketplace comments.

---

## 🚀 Features

- **Fine-Tuned LLM:** Specialized in Turkish e-commerce domain.
- **Dual Analysis:** Simultaneously extracts:
    - **Sentiment:** Positive, Negative, Neutral.
    - **Intent:** Returns, Complaints, General Inquiries, Technical Issues, etc.
- **Efficient Inference:** Uses **4-bit Quantization (BitsAndBytes)** for low VRAM usage.
- **Interactive API:** Fast and robust **FastAPI** backend.
- **Modern UI:** Clean and responsive **React** frontend.

---

## 🚀 Technical Highlights & Optimization

The training pipeline is optimized for high-performance execution on consumer-grade GPUs, specifically tested on **NVIDIA GeForce RTX 4070 Ti Super**.

### 1. 4-bit Quantization (Efficiency)
To fit the 7-billion parameter model into standard VRAM, the system utilizes **BitsAndBytes** with **NF4 (NormalFloat 4)** quantization.
- **VRAM Efficiency:** Reduces memory footprint from ~28GB to under 8GB for inference.
- **Compute:** Leverages `float16` for computation to maintain throughput and precision.

### 2. Parameter-Efficient Fine-Tuning (LoRA)
Instead of updating the entire model, we use **LoRA** to train small adapter layers:
- **Rank ($r=64$):** High rank selected for capturing the nuances of Turkish e-commerce jargon.
- **Alpha ($32$):** Optimized for balanced gradient scaling during backpropagation.
- **Architecture:** Injects trainable rank decomposition matrices into the Transformer layers.

---

## 📊 Training Pipeline

The fine-tuning process follows industry best practices for model robustness:
- **Data Splitting:** Dataset is partitioned into **80% Training**, **10% Validation**, and **10% Test** sets.
- **Overfitting Prevention:** The `SFTTrainer` monitors `eval_loss` and employs **EarlyStopping** logic to save the Best Model based on validation performance.
- **Optimization:** Uses a **Cosine Learning Rate Scheduler** with a warmup phase.

### Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Trendyol-LLM-7b-v1.0 | State-of-the-art Turkish foundation model. |
| **Epochs** | 3 | Optimized number of passes for domain adaptation. |
| **Effective Batch Size** | 8 | (Batch: 4 x Accumulation: 2) |
| **Learning Rate** | 2e-4 | Peak learning rate for AdamW optimizer. |
| **Context Length** | 512 Tokens | Maximum input sequence length. |

---

## 📂 Project Structure

```bash
Proje_LLM/
├── api.py              # FastAPI Backend (Model Inference)
├── scripts/            # Training scripts
│   ├── egitim.py       # Fine-tuning script (LoRA + NF4)
│   ├── degerlendirme.py# Evaluation & Metric generation
│   └── check_setup.py  # Hardware diagnostic tool
├── data/               # Training & Validation datasets
├── models/             # Saved LoRA adapters (Managed via LFS)
├── arayuz/             # React (Vite) Frontend application
└── requirements.txt    # Python dependencies
```

---

## 📥 Getting Started

### 1. Clone the Repository (Git LFS)
This project uses **Git LFS** for large model files. Ensure you have it installed before cloning.

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the project
git clone https://github.com/emreilr/SmartE-CommerceAssistant.git
cd Proje_LLM

# Manually pull LFS files if they are not downloaded automatically
git lfs pull
```

### 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage

**To Train:**
```bash
python scripts/egitim.py
```

**To Run the Full App:**
1. Start the Backend:
   ```bash
   python api.py
   ```
2. Start the Frontend:
   ```bash
   cd arayuz
   npm install
   npm run dev
   ```

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.