# Machine Learning Examples

[![PyPI - Version](https://img.shields.io/pypi/v/mnist.svg)](https://pypi.org/project/mnist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mnist.svg)](https://pypi.org/project/mnist)

-----

## Table of Contents

- [Setup](#setup)
- [Models](#models)
  - [MNIST Digit Classification](#mnist-digit-classification)
  - [Logistic Regression](#logistic-regression)
  - [Linear Regression](#linear-regression)
  - [Content-Based Filtering](#content-based-filtering)
  - [Fraud Detection](#fraud-detection)
  - [Patient Clustering](#patient-clustering)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Vanishing Gradient Problem](#vanishing-gradient-problem)
- [License](#license)

## Project Structure
```
src/
├── clustering/          # Patient clustering analysis
├── fraud-detection/     # Fraud detection models
├── sentiment_analysis/  # Text sentiment analysis
├── mnist/              # MNIST digit classification
├── logistic-regression/ # Binary classification
├── linear-regression/   # Linear regression
├── content-based-filtering/ # Content-based filtering
├── vanishing-gradient/  # Vanishing gradient examples
└── requirements.txt     # Dependencies
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/mineme0110/ml.git
cd ml
```

2. Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Models

### Patient Clustering

Unsupervised learning for patient segmentation based on health metrics.

1. Install dependencies:

```bash
pip install -r src/clustering/requirements.txt
```

2. Run clustering analysis:

```bash
python src/clustering/train.py
```

Features:
- Multiple clustering algorithms:
  - K-means: Groups patients into k clusters
  - DBSCAN: Density-based clustering
  - Hierarchical: Creates cluster hierarchy
- Health metrics analyzed:
  - Age
  - BMI
  - Blood Pressure
  - Glucose Level
  - Cholesterol
  - Heart Rate
- Evaluation metrics:
  - Silhouette Score
  - Calinski-Harabasz Score
- Visualizations:
  - 2D cluster plots
  - Feature distributions

Example output:
```
Training KMEANS clustering...

Clustering Results:
Number of clusters: 3
Silhouette Score: 0.303
Calinski-Harabasz Score: 543.462

Cluster Characteristics:
Cluster 0 (Young, Healthy):
- Age: ~25 years
- BMI: ~22
- Blood Pressure: ~110

Cluster 1 (Middle-aged):
- Age: ~45 years
- BMI: ~28
- Blood Pressure: ~130

Cluster 2 (Elderly):
- Age: ~70 years
- BMI: ~26
- Blood Pressure: ~145
```

### Sentiment Analysis

Text classification for sentiment analysis of movie reviews.

1. Install dependencies:

```bash
pip install -r src/sentiment_analysis/requirements.txt
```

2. Train the model:

```bash
python src/sentiment_analysis/train.py
```

3. Interactive testing:

```bash
python src/sentiment_analysis/interactive_test.py
```

### MNIST Digit Classification

Neural network for MNIST digit classification using PyTorch.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python src/mnist/train.py
```

### Logistic Regression

Binary classification example using scikit-learn's Logistic Regression.

1. Install dependencies:

```bash
pip install -r src/logistic-regression/requirements.txt
```

2. Run the training script:

```bash
python src/logistic-regression/train.py
```

### Linear Regression

Linear regression example using scikit-learn's Linear Regression.

1. Install dependencies:

```bash
pip install -r src/linear-regression/requirements.txt
```

2. Run the training script:

```bash
python src/linear-regression/train.py
```

This will demonstrate linear regression on a generated dataset with visualization of the regression line.

### Content-Based Filtering

Movie recommendation system using content-based filtering.

1. Install dependencies:

```bash
pip install -r src/content-based-filtering/requirements.txt
```

2. Run the recommendation system:

```bash
python src/content-based-filtering/train.py
```

This will demonstrate movie recommendations based on content similarity using a sample movie dataset. The system considers movie genres, actors, and descriptions to make recommendations.

Example output:

```
Getting recommendations for: The Dark Knight
Recommended Movies:
------------------------------------------------------------
1. Iron Man
   Genres: Action, Adventure, Sci-Fi
   Similarity Score: 0.8245
------------------------------------------------------------
2. The Matrix
   Genres: Action, Sci-Fi
   Similarity Score: 0.7856
```

### Fraud Detection

XGBoost-based fraud detection system for identifying fraudulent transactions.

Prerequisites for Mac users:
```bash
# Install OpenMP library (required for XGBoost)
brew install libomp
```

#### Simple Model
A basic fraud detection model with clear separation between normal and fraudulent transactions.

1. Install dependencies:
```bash
pip install -r src/fraud-detection/requirements.txt
```

2. Run the simple fraud detection model:
```bash
python src/fraud-detection/train_simple.py
```

Features:
- Clear separation between classes
- Basic XGBoost parameters
- Perfect for understanding basic fraud detection concepts
- Shows idealized probability distributions

#### Realistic Model
A more sophisticated model that better represents real-world fraud detection scenarios.

1. Install dependencies (if not already installed):
```bash
pip install -r src/fraud-detection/requirements.txt
```

2. Run the realistic fraud detection model:
```bash
python src/fraud-detection/train_realistic.py
```

Features:
- Handles imbalanced classes
- Uses realistic transaction patterns:
  - Transaction amounts
  - Time of day patterns
  - Geographic distances
  - Transaction frequency
- Early stopping and validation
- Feature importance analysis
- More representative probability distributions

Example output:
```
Realistic Model Performance:
ROC AUC Score: 0.9985
Top 5 Most Important Features:
transaction_amount: 0.4532
time_of_day: 0.2876
distance_from_last: 0.1543
transaction_frequency: 0.0892
feature_5: 0.0157
```

### Vanishing Gradient Problem

Demonstrates the vanishing gradient problem in deep neural networks and solutions.

1. Install dependencies:
```bash
pip install -r src/vanishing-gradient/requirements.txt
```

2. Run the demonstration:
```bash
python src/vanishing-gradient/train.py
```

Features:
- Demonstrates vanishing gradients in deep networks
- Implements multiple solutions:
  - ReLU activation
  - Batch Normalization
  - Residual connections (ResNet-style)
  - LSTM gates
- Visualizations:
  - Gradient flow analysis
  - Training metrics comparison
  - Activation distributions

Example output:
```
Training Deep Network...

Standard Network (with Sigmoid):
Epoch 10 - Loss: 2.345 - Gradient Norm: 0.00023 (Vanishing!)

Solutions Comparison:
ReLU Network:
- Loss: 0.456 - Gradient Norm: 0.0892
BatchNorm Network:
- Loss: 0.234 - Gradient Norm: 0.1234
ResNet Style:
- Loss: 0.123 - Gradient Norm: 0.2345
```

## License

This project is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Project Structure
