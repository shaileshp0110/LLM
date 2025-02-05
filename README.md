# Machine Learning Examples

[![PyPI - Version](https://img.shields.io/pypi/v/mnist.svg)](https://pypi.org/project/mnist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mnist.svg)](https://pypi.org/project/mnist)

-----

## Table of Contents

- [Setup](#setup)
- [Models](#models)
  - [MNIST Digit Classification](#mnist-digit-classification)
  - [Logistic Regression](#logistic-regression)
  - [Content-Based Filtering](#content-based-filtering)
- [License](#license)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/mineme0110/mnist.git
cd mnist
```

2. Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Models

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

This will demonstrate logistic regression on a generated dataset with visualization of the decision boundary.

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

## License

This project is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
