# AND Gate Implementation using Perceptron

## Overview

This code implements a **single-neuron perceptron** to learn and predict the behavior of an AND logic gate. The perceptron is trained using a simple supervised learning algorithm to classify binary inputs and produce the correct AND gate output.

## What is an AND Gate?

An AND gate is a basic digital logic gate that outputs `1` (true) only when **both** of its inputs are `1`. Otherwise, it outputs `0` (false).

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 0      |
| 1       | 0       | 0      |
| 1       | 1       | 1      |

## How the Perceptron Works

### 1. Architecture

The perceptron consists of:
- **Two inputs**: $x_1$ and $x_2$ (the binary values)
- **Two weights**: $w_1$ and $w_2$ (learned parameters)
- **One bias**: $b$ (learned parameter)
- **Activation function**: Binary step function

### 2. Forward Propagation

The neuron computes its output in two steps:

**Step 1: Weighted Sum**

$$n = w_1 \cdot x_1 + w_2 \cdot x_2 + b$$

Or in vector notation:

$$n = \mathbf{w}^T \mathbf{x} + b$$

**Step 2: Activation Function**

$$a = f(n) = \begin{cases} 
1 & \text{if } n \geq 0 \\
0 & \text{if } n < 0
\end{cases}$$

where $a$ is the final output of the neuron.

### 3. Training Process (Perceptron Learning Rule)

The perceptron is trained using the following update rules:

**Error Calculation:**

$$e = t - a$$

where:
- $t$ is the target (desired output)
- $a$ is the actual output
- $e$ is the error

**Weight Update:**

$$w_i^{new} = w_i^{old} + e \cdot x_i$$

for each weight $w_i$ corresponding to input $x_i$.

**Bias Update:**

$$b^{new} = b^{old} + e$$

### 4. Training Algorithm

1. **Initialize** weights and bias to zero: $w_1 = 0, w_2 = 0, b = 0$
2. **Set** number of training epochs (e.g., 100)
3. **For each epoch:**
   - For each training example $(x_1, x_2, t)$:
     - Compute output: $a = f(w_1x_1 + w_2x_2 + b)$
     - Calculate error: $e = t - a$
     - Update weights: $w_i = w_i + e \cdot x_i$
     - Update bias: $b = b + e$
4. **Output** the learned weights and bias
5. **Test** the trained perceptron on all inputs

## Code Structure

### Main Components

- **`activationFunction(n)`**: Implements the binary step function
- **`forwardPropagation(p, weights, bias)`**: Computes neuron output for given inputs
- **`main()`**: Orchestrates the training and testing process

### Training Data

```python
P = [[0, 0], [0, 1], [1, 0], [1, 1]]  # All possible input combinations
t = [0, 0, 0, 1]                       # AND gate truth table outputs
```

## Results

After training for 100 epochs, the perceptron learns appropriate weights and bias that allow it to correctly classify all four input combinations according to the AND gate logic. The final weights and bias define a decision boundary that separates the positive class (output 1) from the negative classes (output 0).

## Mathematical Interpretation

The trained perceptron essentially learns a linear decision boundary:

$$w_1 x_1 + w_2 x_2 + b = 0$$

Points where $w_1 x_1 + w_2 x_2 + b \geq 0$ are classified as `1`, and points where $w_1 x_1 + w_2 x_2 + b < 0$ are classified as `0`. This linear separability property makes the AND gate learnable by a single perceptron.
