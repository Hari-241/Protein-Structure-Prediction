# Protein Secondary Structure Prediction (SC4001)

## Overview

This repository contains the source code and documentation for the **SC4001 Neural Networks & Deep Learning** Group Project (Task B). The objective of this project is to predict the secondary structure of proteins based on their amino acid sequences using various Deep Learning architectures.

We tackle two specific classification tasks:

1.  **Q3 Prediction:** Classifying residues into 3 classes: Helix (H), Sheet (E), or Coil (C).
2.  **Q8 Prediction:** Classifying residues into 8 detailed structural classes.

## Repository Structure

The project is divided into three main notebooks to handle different prediction tasks and model architectures:


├── Q3_Structure_Prediction.ipynb   # Models trained from scratch (CNN, BiLSTM, Transformer, Hybrid) for 3-class prediction
├── Q8_Structure_Prediction.ipynb   # Models trained from scratch (CNN, BiLSTM, Transformer, Hybrid) for 8-class prediction
└── ESM_Prediction.ipynb            # Transfer learning using the Pretrained ESM-2 Model (for both Q3 & Q8)
