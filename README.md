# LLM-Based-Sentiment-Classification-and-Embedding-Comparison

This project implements multi-class sentiment classification of tweets using transformer-based models — BERT and FLAN-T5 — with PyTorch and HuggingFace Transformers. It includes tokenization, data preprocessing, model fine-tuning, evaluation, and hyperparameter experimentation on a labeled tweet dataset

Overview
Multi-class sentiment labels: Negative, Neutral, Positive
Models used: bert-base-uncased and flan-t5-base
Frameworks: PyTorch, HuggingFace Transformers
Key techniques: Tokenization, Attention Masking, Learning Rate Scheduling, CNN vs LLM comparison

Key Features
Fine-tuned BERT and FLAN-T5 models for tweet classification
Compared different learning rates and their effects on model convergence
Applied data batching, padding, and attention masking
Used DataCollator and Trainer API for FLAN-T5 training and evaluation
Evaluated model performance using macro-F1 and class-wise F1 scores

Results
BERT Results
Best Macro F1 Score: 0.68 (at Epoch 4)
Performance drops slightly beyond Epoch 3, indicating potential overfitting

FLAN-T5 Results
Class-wise F1 Scores:
Negative: 0.69
Neutral: 0.61
Positive: 0.76
Macro F1 Score: 0.68

How It Works
Tokenization
Used WordPiece tokenizer (BERT) and sequence-to-sequence tokenizer (FLAN-T5)
Handled special tokens like [CLS], [SEP] (BERT) and <pad>, <eos> (FLAN-T5)

Data Loading
Utilized TensorDataset and DataLoader (BERT) and DataCollatorForSeq2Seq (T5)

Applied padding, attention masks, and label encoding for efficient batch processing

Training
BERT: Used AdamW optimizer, learning rate scheduler, and CrossEntropyLoss

T5: Used HuggingFace Trainer and Seq2SeqTrainer for training and evaluation

Hyperparameter Tuning
Experimented with learning rates: 1e-3, 5e-4, 1e-4, 1e-5

Lower learning rates (~1e-4) resulted in smoother convergence and better generalization


Future Improvements
Perform cross-validation to avoid dataset bias
Address class imbalance with sampling techniques
Explore ensemble of BERT and FLAN-T5 predictions
