# GPT-Neo Medical Fine-Tuning using LoRA

This project demonstrates domain adaptation of a Large Language Model (LLM) by fine-tuning **GPT-Neo-125M** on a medical question-answering dataset using **LoRA (Low-Rank Adaptation)**.

The fine-tuned model achieves a significant improvement in performance, reducing perplexity by **97.95%**, demonstrating effective adaptation to the medical domain.

The final model is publicly available on Hugging Face:  
https://huggingface.co/YousefBadr/gptneo-medical-lora

---

# Project Overview

Large Language Models trained on general data often lack domain-specific knowledge. This project addresses that limitation by fine-tuning GPT-Neo for medical question answering using parameter-efficient fine-tuning.

Instead of full fine-tuning, LoRA was used to efficiently adapt the model while training less than 1% of total parameters.

This approach provides:

- Faster training
- Lower GPU memory usage
- Reduced computational cost
- Strong domain performance improvement

---

# Dataset

Dataset: Medical Meadow Medical Flashcards  
https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

Dataset characteristics:

- ~34,000 medical question-answer samples
- English language
- Instruction-based format:

```
Instruction: Answer this question truthfully
Question: <medical question>
Response: <medical answer>
```

Split:

- Training: 90%
- Validation: 10%

---

# Model Architecture

Base model: GPT-Neo-125M

Architecture details:

- Transformer decoder architecture
- 12 layers
- 12 attention heads
- Hidden size: 768
- 125 million parameters

Fine-tuning method: LoRA

LoRA configuration:

- Rank: 16
- Alpha: 32
- Dropout: 0.1

---

# Training Configuration

Hardware:

- NVIDIA Tesla T4 GPU
- Google Colab environment

Training parameters:

- Batch size: 8
- Learning rate: 2e-4
- Optimizer: AdamW
- Precision: FP16
- Training method: LoRA fine-tuning
- LoRA weights merged into base model after training

---

# Evaluation Results

Evaluation metric: Perplexity

Perplexity measures how well the model predicts text. Lower values indicate better performance.

Results:

| Model | Loss | Perplexity |
|------|------|------------|
| Base GPT-Neo-125M | 4.7494 | 115.51 |
| GPT-Neo-125M + LoRA | 0.8638 | 2.37 |

Performance improvement:

Perplexity reduced by **97.95%**

This demonstrates successful domain adaptation and significant improvement in medical knowledge generation.

---

# Example Comparison

Input:

```
Instruction: Answer this question truthfully
Question: What causes asthma?
Response:
```

Base model output:

Generic response with limited medical accuracy.

Fine-tuned model output:

Accurate medical explanation using correct clinical terminology.

---

# Project Structure

```
.
├── README.md
├── training_notebook.ipynb
├── training_notebook.py
├── model_performance_comparison.csv
├── generation_comparison.csv
├── perplexity_chart.png
└── loss_chart.png
```

---

# How to Use the Model

Install dependencies:

```
pip install transformers torch
```

Load and use the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("YousefBadr/gptneo-medical-lora")
tokenizer = AutoTokenizer.from_pretrained("YousefBadr/gptneo-medical-lora")

prompt = """Instruction: Answer this question truthfully
Question: What causes diabetes?
Response:"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=200
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

# Technical Skills Demonstrated

- Large Language Model fine-tuning
- LoRA parameter-efficient fine-tuning
- Hugging Face Transformers and PEFT
- PyTorch training and inference
- Dataset preprocessing and tokenization
- Model evaluation using perplexity
- Model deployment on Hugging Face
- GPU training optimization

---

# Limitations

- Model may generate incorrect medical information
- Not suitable for clinical or diagnostic use
- Limited by dataset size and scope
- Intended for research and educational purposes only

---

# Hugging Face Model

Model available at:

https://huggingface.co/YousefBadr/gptneo-medical-lora

---

# Author

Yousef Badr  
Machine Learning Engineer  

Hugging Face:  
https://huggingface.co/YousefBadr

---

# Acknowledgment

Special thanks to Eng. Mahmoud Khorshid for guidance and support throughout this project.

---

# License

Apache 2.0
