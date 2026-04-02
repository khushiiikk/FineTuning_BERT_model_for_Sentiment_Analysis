Transforming pre-trained BERT into a sentiment classification powerhouse with custom neural architecture

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="https://raw.githubusercontent.com/google-research/bert/master/bert_diagrams.png" width="600" alt="BERT Architecture"/>
</p>

## 🚀 What This Project Does

This project demonstrates **transfer learning** at its finest—taking Google's pre-trained BERT (Bidirectional Encoder Representations from Transformers) and fine-tuning it for binary sentiment classification. Instead of training a massive model from scratch (hello, overfitting! 👋), we leverage BERT's pre-trained knowledge and add custom layers on top.

**The Magic:** Freeze BERT's 110M parameters, train only our lightweight classifier head. Efficient. Fast. Effective.

## 🧠 Architecture Overview
┌─────────────────────────────────────────────────────────┐
│                    INPUT TEXT                           │
│              "This movie was absolutely                 │
│                  fantastic and thrilling!"              │
└──────────────────────┬──────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│              BERT TOKENIZER                             │
│         (WordPiece → IDs + Attention Masks)             │
└──────────────────────┬──────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────┐    │
│  │  BERT-BASE-UNCASED (FROZEN ❄️)                  │    │
│  │  • 12 Transformer Layers                        │    │
│  │  • 768 Hidden Dimensions                        │    │
│  │  • 12 Attention Heads                           │    │
│  │  • 110M Parameters (NOT TRAINED)                │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │ [CLS] Token (768-dim)             │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │  CUSTOM CLASSIFIER HEAD (TRAINABLE 🔥)          │    │
│  │  ┌─────────────┐    ┌─────────────┐   ┌──────┐ │    │
│  │  │  Linear     │───→│   ReLU      │──→│ Drop │ │    │
│  │  │  768 → 512  │    │ Activation  │   │ 0.2  │ │    │
│  │  └─────────────┘    └─────────────┘   └──┬───┘ │    │
│  │                                          │      │    │
│  │  ┌─────────────┐    ┌────────────────┐   │      │    │
│  │  │  Linear     │←───│   LogSoftmax   │←──┘      │    │
│  │  │  512 → 2    │    │   (Output)     │          │    │
│  │  └─────────────┘    └────────────────┘          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
▼
┌─────────────────┐
│  SENTIMENT      │
│  [0.02, 0.98]   │
│  → POSITIVE ✅  │
└─────────────────┘
plain
Copy

## 📊 Dataset Structure

| Column | Description | Values |
|--------|-------------|--------|
| `sentence` | Raw text data | String |
| `label` | Sentiment class | `0` = Negative 😞, `1` = Positive 😊 |

**Split Strategy:** 70% Train | 15% Validation | 15% Test (Stratified)

## ⚡ Quick Start

### Prerequisites
```bash
pip install transformers torch numpy pandas scikit-learn matplotlib
