# Transformer From Scratch 

This repository provides a fully custom implementation of the Transformer architecture using PyTorch — from the ground up. It is designed for educational purposes, model experimentation, and visualizations using Streamlit.

## Overview

This project includes:

- A **Transformer** architecture implemented from scratch.
- Support for **classification** and **regression** heads.
- A **Streamlit app** to visualize and interact with attention mechanisms.

---

## Architecture Highlights

- **Positional Encoding**
- **Multi-Head Attention**
- **Encoder/Decoder Blocks**
- **Feedforward Layers**
- **Normalization & Residuals**
- **Custom Heads** for classification or regression

---

## Directory Structure

```
Transformer_From_Scratch/
│
├── model/                 
│   ├── tokenizer.py.py
│   ├── attention.py
│   ├── attention_utils.py
│   └── transformer.py
│
├── app.py
│
├── notebook/
│   └── transformer_scratch.ipynb
│
└── README.md
```

---

## Streamlit Visualization

We’ve included a lightweight Streamlit app to visualize multi-head attention from your Transformer model.

###  Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
---

##  Training & Evaluation

- This repo is focused on architecture and visualization and trained on toy dataset.
- If you'd like to include training data (data/train.txt), let us know or contribute via pull request.

---

##  Inspiration

Inspired by the original [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), this repo aims to demystify Transformers through readable code and interactive demos.

---

##  Contributing

Contributions are welcome! If you have ideas for improving the visualization, optimization, or training loop — feel free to open an issue or submit a PR.

---

