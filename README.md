# Hotel Review Sentiment Analysis using Deep Learning

## 📝 Project Overview
This project implements and compares deep learning approaches for sentiment analysis of hotel reviews using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The system classifies hotel reviews into 5-star rating categories (1-5 stars) based on textual content.

## 🎯 Objectives
- Develop automated sentiment classification for hotel reviews
- Compare performance between Simple RNN and LSTM architectures
- Investigate the impact of Word2Vec embeddings vs. random embeddings
- Address class imbalance in review rating distributions

## 📊 Dataset
- **Source**: Hotel_review.csv from university AI/ML module resources
- **Total Reviews**: 20,491 hotel reviews with ratings 1-5
- **Distribution**:
  - Rating 1: 1,421 reviews (6.9%)
  - Rating 2: 1,793 reviews (8.7%)
  - Rating 3: 2,184 reviews (10.7%)
  - Rating 4: 6,039 reviews (29.5%)
  - Rating 5: 9,054 reviews (44.2%)
- **Train/Validation Split**: 80/20 stratified split
  - Training: ~16,393 reviews
  - Validation: ~4,098 reviews

## 🔧 Technologies Used
- **Python 3.x**: Programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy & Pandas**: Data manipulation
- **NLTK**: Text preprocessing
- **Word2Vec**: Pre-trained embeddings
- **Scikit-learn**: Evaluation metrics
- **Matplotlib/Seaborn**: Visualization

## 🏗️ Model Architectures

### 1. Simple RNN Model
- Bidirectional SimpleRNN with 64 units
- Embedding layer with L2 regularization
- Dropout layers (40%) for regularization
- GlobalMaxPooling1D for dimensionality reduction
- Dense layers with ReLU and Softmax activations
- **Achieved**: 61.89% test accuracy

### 2. LSTM Model
- Two stacked Bidirectional LSTM layers (64 units each)
- SpatialDropout1D and standard Dropout
- Layer normalization for training stability
- L2 regularization and recurrent dropout
- **Achieved**: 56.30% test accuracy (balanced dataset)

### 3. Word2Vec Enhanced LSTM
- Pre-trained Google News Word2Vec embeddings (300 dimensions)
- 85.65% vocabulary coverage
- Two-phase training: frozen embeddings → fine-tuning
- Custom classification head with multiple dense layers
- **Achieved**: 56.30% test accuracy

## 📈 Results Summary

### Model Validation Metrics
| Model                 | Test Accuracy | Key Strengths                              |
|-----------------------|---------------|--------------------------------------------|
| Simple RNN            | 61.89%        | Better performance on extreme sentiments   |
| LSTM + Word2Vec       | 56.30%        | More balanced precision/recall across classes |

### Class-Specific Performance (Word2Vec LSTM)
| Rating                  | F1-Score |
|-------------------------|----------|
| Rating 1 (Very Negative) | 0.71     |
| Rating 2 (Negative)      | 0.49     |
| Rating 3 (Neutral)       | 0.43     |
| Rating 4 (Positive)      | 0.47     |
| Rating 5 (Very Positive) | 0.70     |

## 🔍 Key Findings
- **Sentiment Extremity Pattern**: Models performed significantly better on extreme sentiments (ratings 1 & 5) than neutral ratings
- **Word2Vec Impact**: Pre-trained embeddings provided modest improvements but didn't dramatically enhance performance
- **Class Imbalance**: Original dataset heavily skewed toward positive reviews required balanced sampling
- **Adjacent Confusion**: Most misclassifications occurred between neighboring rating classes
