# Protein Secondary Structure Prediction

A deep learning project comparing multiple neural network architectures for predicting protein secondary structures at both Q3 (3-class) and Q8 (8-class) levels.

## 📋 Project Overview

This project implements and evaluates five different deep learning approaches for protein secondary structure prediction:

- **CNN-Based Model** - Captures local sequence patterns using 1D convolutions with residual connections
- **BiLSTM Model** - Learns bidirectional sequential dependencies
- **Transformer Model** - Models long-range dependencies using self-attention
- **Hybrid CNN-BiLSTM-Transformer** - Combines strengths of all three architectures
- **ESM-Based Model** - Leverages pretrained protein language model embeddings (ESM2)

### Tasks
- **Q3 Prediction**: Classifies residues into 3 classes (Helix, Sheet, Coil)
- **Q8 Prediction**: Classifies residues into 8 detailed structural classes

## 🎯 Results

### Q3 (3-Class) Performance

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|--------------|
| CNN | 70.70% | 71.00% |
| BiLSTM | 59.31% | 59.62% |
| Transformer | 50.04% | N/A |
| Hybrid | 68.46% | 69.60% |
| **ESM (Best)** | **81.60%** | **82.04%** |

### Q8 (8-Class) Performance

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|--------------|
| CNN | 52.49% | 55.00% |
| BiLSTM | 46.30% | 46.87% |
| Transformer | 39.79% | 39.79% |
| Hybrid | 57.69% | 58.00% |
| **ESM (Best)** | **70.30%** | **70.45%** |

## 📁 Repository Structure

```
.
├── Q3_Prediction.ipynb          # Q3 structure prediction (CNN, BiLSTM, Transformer, Hybrid)
├── Q8_Prediction.ipynb          # Q8 structure prediction (CNN, BiLSTM, Transformer, Hybrid)
├── ESM_Prediction.ipynb         # ESM-based prediction for both Q3 and Q8
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install fair-esm  # For ESM-based models
pip install optuna    # For hyperparameter tuning
```

### Dataset

The project uses the **2018-06-06 PDB Intersect Pisces dataset** derived from the Protein Data Bank (PDB). Each entry contains:
- `seq` - Amino acid sequence (A-Y)
- `sst3` - Q3 secondary structure labels
- `sst8` - Q8 secondary structure labels
- `len` - Sequence length

Data split: 70% training, 15% validation, 15% testing

## 📓 Notebooks

### 1. Q3_Prediction.ipynb
Contains implementations and training for Q3 prediction using:
- CNN with residual blocks and dilated convolutions
- BiLSTM with bidirectional processing
- Transformer encoder with multi-head attention
- Hybrid architecture combining CNN + BiLSTM + Transformer

### 2. Q8_Prediction.ipynb
Same architectures as Q3 but configured for 8-class prediction with adjusted hyperparameters and training strategies.

### 3. ESM_Prediction.ipynb
Implements the ESM2-based approach:
- Uses pretrained `esm2_t33_650M_UR50D` model for embeddings
- Trains lightweight classifier on frozen ESM embeddings
- Separate models for Q3 and Q8 tasks
- Achieves best performance across both tasks

## 🔧 Model Architectures

### CNN Model
- Embedding layer with positional encoding
- Residual 1D convolution blocks with dilations
- Batch normalization and dropout
- Layer normalization before output

### BiLSTM Model
- Embedding layer
- Bidirectional LSTM
- Packed sequences for efficient padding handling
- Linear output layer

### Transformer Model
- Embedding + positional encoding
- Multi-head self-attention (6 heads, 3 layers)
- Feed-forward networks
- Padding mask for attention

### Hybrid Model
- CNN block for local features (192 → 128 → 192 channels)
- BiLSTM for contextual features (hidden size: 256)
- Transformer encoder for long-range dependencies (3 layers, 6 heads)
- Dropout + linear output layer

### ESM-Based Model
- Frozen ESM2 embeddings (650M parameters)
- Lightweight classifier: BiLSTM + attention + linear layers
- Only classifier trained (~2-5M parameters)

## 🎓 Training Details

### Common Setup
- **Optimizer**: Adam/AdamW
- **Loss**: Cross-entropy (ignoring padding)
- **Batch Size**: 64
- **Hardware**: CUDA GPU
- **Techniques**: 
  - Mixed precision training
  - Gradient clipping
  - Early stopping
  - Learning rate scheduling

### Model-Specific
- **CNN**: Learning rate 1.6e-04, ReduceLROnPlateau scheduler
- **Hybrid**: Learning rate 4.7e-03, OneCycleLR scheduler, label smoothing
- **ESM**: Learning rate 4.1e-04, ReduceLROnPlateau, 30 epochs max

## 📊 Key Findings

1. **ESM dominates**: Pretrained protein language models provide the strongest performance for both Q3 and Q8 tasks
2. **CNN performs well**: Local pattern recognition through convolutions is effective for secondary structure
3. **Hybrid improves over baselines**: Combining multiple architectures captures complementary features
4. **Transformers struggle**: Limited data and model size hindered standalone transformer performance
5. **BiLSTM underperforms**: Sequential processing alone is insufficient for this task

## 🔬 Technical Highlights

- **Proper padding handling**: Binary masks ensure loss computation ignores padded positions
- **Hyperparameter tuning**: Optuna used for hybrid model optimization
- **Data preprocessing**: Integer encoding, sequence padding, length-based filtering
- **Evaluation**: Per-residue accuracy, confusion matrices, class-wise performance

## 👥 Authors

- Ananthula Harineesh Reddy (U2323714G)
- Natraj Kalingarayar Amogh Sriman Kalinganayar (U2323933F)
- Sunilkumar Hrishikesh (U2323903A)

**Institution**: Nanyang Technological University  
**Course**: SC4001 Neural Networks and Deep Learning

## 📚 References

1. European Bioinformatics Institute (EBI). [Introduction to Protein Structure: Secondary Structure](https://www.ebi.ac.uk/training/online/courses/introduction-protein-structure/secondary-structure/)
2. Rives A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15): e2016239118
3. Meta AI Research. [ESM Protein Language Models](https://github.com/facebookresearch/esm)
4. [Protein Data Bank (PDB)](https://www.rcsb.org/)

## 📄 License

This project is part of academic coursework at NTU.

##  Acknowledgments

- Meta AI for the ESM2 pretrained models
- Protein Data Bank for the structural dataset
- NTU SC4001 course staff for guidance
