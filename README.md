# IA - Artificial Intelligence Labs

Repository containing implementation and documentation for various AI and Machine Learning algorithms.

## Repository Structure

### Lab 1 & 2: Machine Learning Fundamentals
Located in `Lab1_2/`

#### Implementations
- **[Nearest Neighbor & K-Nearest Neighbors](Lab1_2/NN_KNN.py)** - Instance-based learning classifiers
- **[Linear Classifier with SVM Loss](Lab1_2/LinearClassifierWithSVMLoss.py)** - Multi-class classification using hinge loss

#### Documentation
- **[NN & KNN Documentation](Lab1_2/NN_KNN.md)** - Comprehensive guide on Nearest Neighbor and K-Nearest Neighbors algorithms, CIFAR-10 dataset, distance metrics, and performance analysis
- **[SVM Loss Documentation](Lab1_2/LinearClassifierWithSVMLoss.md)** - Detailed explanation of linear classifiers, SVM hinge loss, gradient computation, and training algorithms

### Lab 3: Neural Networks
Located in `Lab3/`

#### Implementations
- **[AND Gate Perceptron](Lab3/AND.py)** - Single-neuron perceptron for learning AND logic gate
- **[XOR Gate Neural Network](Lab3/XOR.py)** - Multi-layer network with backpropagation for learning XOR logic gate

#### Documentation
- **[AND Gate Documentation](Lab3/AND.md)** - Explanation of perceptron architecture, forward propagation, perceptron learning rule, and binary step activation function
- **[XOR Gate Documentation](Lab3/XOR.md)** - Complete guide on multi-layer neural networks, backpropagation algorithm, sigmoid activation, gradient descent, and non-linear separability

## Setup

### Prerequisites
- Python 3.x installed

### Installation

Run the setup script to create a virtual environment and install all dependencies:

```powershell
.\setup.ps1
```

Or manually:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- numpy
- opencv-python
- tensorflow
- keras
- matplotlib

## Quick Start

### Run AND Gate Perceptron
```powershell
python Lab3\AND.py
```

### Run XOR Neural Network
```powershell
python Lab3\XOR.py
```

### Run Linear Classifier with SVM Loss
```powershell
python Lab1_2\LinearClassifierWithSVMLoss.py
```

### Run Nearest Neighbor Classification
```powershell
python Lab1_2\NN_KNN.py
```

## Key Concepts Covered

### Machine Learning Basics
- Instance-based learning (NN, KNN)
- Distance metrics (L1, L2)
- Linear classification
- Support Vector Machines (SVM)
- Hinge loss and margin optimization
- Gradient descent

### Neural Networks
- Perceptron architecture
- Activation functions (binary step, sigmoid)
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Multi-layer networks
- Linear vs. non-linear separability

## License

This repository is for educational purposes.
