# 🎬 BERT Fine-Tuning for Sentiment Classification (IMDB)

This project fine-tunes a **BERT model** (`bert-base-uncased`) on the **IMDB movie reviews dataset** for binary **sentiment classification** (positive / negative).  
It uses **Hugging Face Transformers**, **Datasets**, and **Accelerate** to enable **GPU and mixed-precision training** for faster, memory-efficient fine-tuning.

---

## 🧠 Project Overview

This script demonstrates an end-to-end NLP workflow:
1. Load and preprocess text data from the IMDB dataset.
2. Tokenize text using BERT’s WordPiece tokenizer.
3. Fine-tune the pretrained BERT model on sentiment labels.
4. Evaluate model performance with accuracy and F1-score.
5. Save and run inference using Hugging Face’s `pipeline()`.

---

## Prerequisites

python==3.11

## ⚙️ Features
- ✅ GPU / mixed-precision (fp16 or bf16) support  
- ✅ Automatic dataset tokenization and padding  
- ✅ Metric tracking (Accuracy, F1)  
- ✅ Model checkpoint saving & best model selection  
- ✅ Ready-to-use inference pipeline  
- ✅ Modern `processing_class=` API compatibility (Transformers ≥ 4.57)

---

## 🧩 Installation

Step 1: Clone repository

```
git clone https://github.com/Montreal-JieZhu/BERT_SFT_Sentiment_Classification.git
cd BERT_SFT_Sentiment_Classification
```

Step 2: Setup python environment

If you have **uv  -- one cmd done**

```
uv sync
```

Or

```
pip install requirements.txt
```

A GPU (CUDA) is strongly recommended for training.

---

## 🚀 Training

The script automatically:
- Detects GPU and sets mixed-precision (`fp16` or `bf16`)
- Fine-tunes BERT for 3 epochs
- Evaluates after each epoch
- Saves the best model to `bert-imdb/best`

---

## 🧪 Inference Example

After training, test the model interactively:

```python
from transformers import pipeline
clf = pipeline("text-classification", model="bert-imdb/best", tokenizer="bert-imdb/best", device_map="auto")

print(clf("This movie was absolutely wonderful!"))
print(clf("Terrible plot and wooden acting."))
```

Output example:
```
[{'label': 'POSITIVE', 'score': 0.99}]
[{'label': 'NEGATIVE', 'score': 0.98}]
```

---

## 📈 Results

| Metric | Score (approx.) |
|:-------|:----------------|
| Accuracy | ~94% |
| F1 Score | ~94% |

*(Exact results depend on random seed and hardware.)*

---

## 🧰 Notes
- Adjust `max_length`, batch size, or learning rate as needed.  
- For smaller GPUs, enable gradient checkpointing or reduce batch size.  

---

## 🪪 License
This project is released under the **MIT License**.  
You’re free to use, modify, and distribute with attribution.

---
