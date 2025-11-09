# XOR Gate Implementation using Multi-Layer Neural Network

## Overview

This code implements a **multi-layer neural network** with backpropagation to learn and predict the behavior of an XOR logic gate. Unlike the simple perceptron used for the AND gate, XOR requires a hidden layer because it is not linearly separable.

## What is an XOR Gate?

An XOR (exclusive OR) gate is a digital logic gate that outputs `1` (true) when the inputs are **different**, and `0` (false) when the inputs are the same.

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## Why XOR Needs Multiple Layers

XOR is **not linearly separable**, meaning no single straight line can separate the outputs. This is why a single perceptron cannot learn XOR. We need a hidden layer to transform the input space into a representation where the classes become linearly separable.

## Network Architecture

The neural network consists of:
- **Input Layer**: 2 neurons ($x_1$, $x_2$)
- **Hidden Layer**: 2 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

```
Input Layer (2) → Hidden Layer (2) → Output Layer (1)
```

## Mathematical Foundation

### 1. Activation Function

**Sigmoid Function:**

$$\sigma(n) = \frac{1}{1 + e^{-n}}$$

**Properties:**
- Output range: $(0, 1)$
- Smooth and differentiable
- Used for both hidden and output layers

**Derivative of Sigmoid:**

$$\sigma'(n) = \sigma(n) \cdot (1 - \sigma(n))$$

This derivative is crucial for backpropagation.

### 2. Forward Propagation

**Hidden Layer Computation:**

For each neuron $j$ in the hidden layer:

$$n_j^{(1)} = \sum_{i=1}^{2} w_{ij}^{(1)} x_i + b_j^{(1)}$$

$$a_j^{(1)} = \sigma(n_j^{(1)})$$

In matrix form:

$$\mathbf{n}^{(1)} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}$$

$$\mathbf{a}^{(1)} = \sigma(\mathbf{n}^{(1)})$$

where:
- $\mathbf{X}$ is the input matrix (4×2)
- $\mathbf{W}^{(1)}$ is the weight matrix for layer 1 (2×2)
- $\mathbf{b}^{(1)}$ is the bias vector for layer 1 (1×2)
- $\mathbf{a}^{(1)}$ is the hidden layer output (4×2)

**Output Layer Computation:**

$$n^{(2)} = \sum_{j=1}^{2} w_j^{(2)} a_j^{(1)} + b^{(2)}$$

$$a^{(2)} = \sigma(n^{(2)})$$

In matrix form:

$$\mathbf{n}^{(2)} = \mathbf{a}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

$$\mathbf{a}^{(2)} = \sigma(\mathbf{n}^{(2)})$$

where:
- $\mathbf{W}^{(2)}$ is the weight matrix for layer 2 (2×1)
- $\mathbf{b}^{(2)}$ is the bias vector for layer 2 (1×1)
- $\mathbf{a}^{(2)}$ is the final output (4×1)

### 3. Cost Function

The network uses **Mean Squared Error (MSE)** as the cost function:

$$C = \frac{1}{2m} \sum_{i=1}^{m} (t_i - a_i^{(2)})^2$$

where:
- $m$ is the number of training examples (4 for XOR)
- $t_i$ is the target output
- $a_i^{(2)}$ is the predicted output

### 4. Backpropagation

Backpropagation computes the gradients of the cost function with respect to weights and biases.

**Output Layer Gradients:**

Error at output layer:

$$\mathbf{E}^{(2)} = \mathbf{t} - \mathbf{a}^{(2)}$$

Gradient (delta) for output layer:

$$\boldsymbol{\delta}^{(2)} = \mathbf{E}^{(2)} \odot \sigma'(\mathbf{a}^{(2)})$$

where $\odot$ denotes element-wise multiplication.

**Hidden Layer Gradients:**

Error propagated to hidden layer:

$$\mathbf{E}^{(1)} = \boldsymbol{\delta}^{(2)} (\mathbf{W}^{(2)})^T$$

Gradient (delta) for hidden layer:

$$\boldsymbol{\delta}^{(1)} = \mathbf{E}^{(1)} \odot \sigma'(\mathbf{a}^{(1)})$$

### 5. Weight and Bias Updates

Using gradient descent with learning rate $\eta$:

**Output Layer Updates:**

$$\mathbf{W}^{(2)} := \mathbf{W}^{(2)} + \eta \cdot (\mathbf{a}^{(1)})^T \boldsymbol{\delta}^{(2)}$$

$$\mathbf{b}^{(2)} := \mathbf{b}^{(2)} + \eta \cdot \sum_{i} \boldsymbol{\delta}_i^{(2)}$$

**Hidden Layer Updates:**

$$\mathbf{W}^{(1)} := \mathbf{W}^{(1)} + \eta \cdot \mathbf{X}^T \boldsymbol{\delta}^{(1)}$$

$$\mathbf{b}^{(1)} := \mathbf{b}^{(1)} + \eta \cdot \sum_{i} \boldsymbol{\delta}_i^{(1)}$$

## Training Algorithm

1. **Initialize** weights and biases randomly from uniform distribution $U(0, 1)$
2. **Set** hyperparameters:
   - Number of epochs: 5000
   - Learning rate: $\eta = 0.3$
3. **For each epoch:**
   
   a. **Forward Propagation:**
      - Compute hidden layer output: $\mathbf{a}^{(1)} = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
      - Compute output layer: $\mathbf{a}^{(2)} = \sigma(\mathbf{a}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$
   
   b. **Backpropagation:**
      - Calculate output layer error: $\mathbf{E}^{(2)} = \mathbf{t} - \mathbf{a}^{(2)}$
      - Calculate output layer gradient: $\boldsymbol{\delta}^{(2)} = \mathbf{E}^{(2)} \odot \sigma'(\mathbf{a}^{(2)})$
      - Propagate error to hidden layer: $\mathbf{E}^{(1)} = \boldsymbol{\delta}^{(2)} (\mathbf{W}^{(2)})^T$
      - Calculate hidden layer gradient: $\boldsymbol{\delta}^{(1)} = \mathbf{E}^{(1)} \odot \sigma'(\mathbf{a}^{(1)})$
   
   c. **Update Parameters:**
      - Update all weights and biases using computed gradients
   
   d. **Compute Cost:**
      - Calculate MSE: $C = \frac{1}{2m} \sum (t_i - a_i^{(2)})^2$
   
   e. **Early Stopping:**
      - If $C < 0.01$, stop training

4. **Output** learned parameters and test predictions

## Code Structure

### Main Functions

- **`sigmoid(n)`**: Implements the sigmoid activation function $\sigma(n) = \frac{1}{1 + e^{-n}}$
- **`sigmoidDerivative(n)`**: Computes the derivative $\sigma'(n) = n(1-n)$
- **`forwardPropagationLayer(p, weights, biases)`**: Computes forward pass for a single layer
- **`main()`**: Orchestrates training and testing

### Training Data

```python
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # All input combinations
labels = np.array([[0], [1], [1], [0]])               # XOR truth table
```

### Network Parameters

```python
inputSize = 2           # Two binary inputs
noNeuronsLayer1 = 2     # Two neurons in hidden layer
noNeuronsLayer2 = 1     # One output neuron
learningRate = 0.3      # Step size for gradient descent
noEpochs = 5000         # Maximum training iterations
```

## Results Interpretation

After training, the network learns:
1. **Hidden layer weights and biases** that transform the input space
2. **Output layer weights and biases** that combine hidden features

The hidden layer creates a representation where:
- One neuron might learn to detect when inputs are different
- Another neuron might learn complementary features
- Together, they make the XOR function linearly separable in the hidden space

## Comparison with AND Gate

| Aspect | AND Gate (Perceptron) | XOR Gate (Multi-Layer) |
|--------|----------------------|------------------------|
| Architecture | Single layer | Two layers (hidden + output) |
| Activation | Binary step function | Sigmoid function |
| Learning | Perceptron rule | Backpropagation |
| Linear Separability | Yes | No (requires transformation) |
| Complexity | Simple | More complex |

## Key Insights

1. **Non-linearity**: The sigmoid activation function introduces non-linearity, allowing the network to learn complex patterns
2. **Hidden Representations**: The hidden layer learns useful feature representations that make the problem solvable
3. **Gradient Descent**: Backpropagation efficiently computes gradients for all parameters simultaneously
4. **Universal Approximation**: With enough hidden neurons, this architecture can approximate any continuous function

## Visualization of Decision Boundary

The trained network creates a non-linear decision boundary in the 2D input space that correctly separates the XOR classes. The hidden layer essentially "folds" the input space so that the output layer can draw a linear separator in the transformed space.
