# Multi-Modal AI for Predicting YouTube Video Success
## Overview

This MSc dissertation project explores a multimodal deep learning approach to predicting YouTube video success based on view count categories. The task is formulated as a 5-class classification problem using both visual and textual content from videos.

Rather than relying solely on metadata, this work integrates frame-level visual features and transcript-based textual representations to improve predictive performance and interpretability.

## Dataset

500 YouTube videos across 50 genres
View categories:

0–10k
10k–100k
100k–500k
500k–1M
1M+

Train/Validation split: 400 / 100
The dataset includes titles, descriptions, transcripts, engagement metrics, and extracted video frames.

## Methodology

### Visual Model (CNN–RNN Architecture)

Frame extraction at 1-second intervals (OpenCV)
ResNet-18 (pretrained on ImageNet) for spatial feature extraction
3-layer GRU (128 hidden units) for temporal modelling
Fully connected classification layer
Cross-Entropy loss with class weighting

### Text Model (BERT)

Pretrained BERT fine-tuned on video transcripts
CLS token embedding used for classification
Focal Loss applied to address class imbalance

### Multimodal Fusion

Late Fusion
Weighted combination of visual and textual logits (α = 0.5)

Attention-Based Fusion
Learned modality weights via fully connected attention mechanism
Precision-Recall evaluation across classes

## Explainability

To improve interpretability and trust in model predictions, the following methods were implemented:

Grad-CAM++
Integrated Gradients
LIME (for text and fusion outputs)
Attention weight visualisation

These techniques provided insight into:

Spatial focus areas in frames
Influential transcript tokens
Modality contributions during fusion

## Key Findings

Text-based modelling slightly outperformed visual modelling in accuracy.
Multimodal fusion improved class-level balance but remained sensitive to dataset imbalance.
Explainability techniques revealed modality bias toward dominant classes.
Visual modelling remains challenging due to temporal complexity and feature variability.
