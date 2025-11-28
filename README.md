Protein Secondary Structure Prediction (SC4001)

Overview

This repository contains the source code and documentation for the SC4001 Neural Networks & Deep Learning Group Project (Task B). The objective of this project is to predict the secondary structure of proteins based on their amino acid sequences using various Deep Learning architectures.

We tackle two specific classification tasks:

Q3 Prediction: Classifying residues into 3 classes: Helix (H), Sheet (E), or Coil (C).

Q8 Prediction: Classifying residues into 8 detailed structural classes.

Repository Structure

The project is divided into three main notebooks to handle different prediction tasks and model architectures:

├── Q3_Structure_Prediction.ipynb   # Models trained from scratch (CNN, BiLSTM, Transformer, Hybrid) for 3-class prediction
├── Q8_Structure_Prediction.ipynb   # Models trained from scratch (CNN, BiLSTM, Transformer, Hybrid) for 8-class prediction
└── ESM_Prediction.ipynb            # Transfer learning using the Pretrained ESM-2 Model (for both Q3 & Q8)


Dataset

Source: 2018-06-06 PDB Intersect Pisces dataset.

Features: Amino acid sequences (mapped to integers or raw strings for ESM).

Preprocessing: Sequences were filtered, padded to uniform lengths, and split into Train (70%), Validation (15%), and Test (15%) sets.

Models Implemented

We implemented and compared five different approaches:

1. CNN (Convolutional Neural Network)

Focuses on capturing local patterns and short-range motifs using 1D convolutions and residual blocks.

2. BiLSTM (Bidirectional LSTM)

Captures sequential context by processing the amino acid chain in both forward and backward directions.

3. Transformer

Utilizes self-attention mechanisms to model long-range dependencies across the entire sequence.

4. Hybrid Model

A combined architecture leveraging the strengths of all three base models:

CNN for local features.

BiLSTM for sequence context.

Transformer for global attention.

5. ESM-2 (Evolutionary Scale Modeling)

Uses a pretrained Protein Language Model (frozen embeddings) with a lightweight classifier trained on top. This approach leverages evolutionary information learned from millions of protein sequences.

Experimental Results

Below is a summary of the accuracy achieved on the Test set for both tasks. The ESM-based model achieved the highest performance, followed by the Hybrid model.

Model

Q3 Accuracy

Q8 Accuracy

CNN

71.00%

55.00%

BiLSTM

59.62%

46.87%

Transformer

N/A

39.79%

Hybrid (CNN+RNN+Attn)

69.60%

58.00%

ESM (Pretrained)

82.04%

70.45%

Getting Started

Prerequisites

To run these notebooks, you will need Python installed along with the following libraries:

torch (PyTorch)

numpy

pandas

sklearn

transformers (for ESM)

matplotlib / seaborn (for plotting)

Installation

pip install torch numpy pandas scikit-learn transformers matplotlib seaborn


Usage

For Q3 Baselines: Open Q3_Structure_Prediction.ipynb to train/test the CNN, BiLSTM, and Hybrid models on the 3-class task.

For Q8 Baselines: Open Q8_Structure_Prediction.ipynb to train/test the CNN, BiLSTM, and Hybrid models on the 8-class task.

For SOTA Results: Open ESM_Prediction.ipynb to run the pretrained ESM-2 embedding extraction and classification.

Authors

Ananthula Harineesh Reddy

Natraj Kalingarayar Amogh Sriman Kalinganayar

Sunilkumar Hrishikesh

Nanyang Technological University, Singapore
