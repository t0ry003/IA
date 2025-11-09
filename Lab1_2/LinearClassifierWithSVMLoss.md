# Linear Classifier with SVM Loss (Hinge Loss)

## Overview

This code implements a **linear multi-class classifier** using **Support Vector Machine (SVM) loss**, also known as **hinge loss**. The classifier learns to separate data into multiple classes by optimizing a weight matrix through gradient descent, ensuring that correct classes are scored higher than incorrect ones with a specified margin.

## What is a Linear Classifier?

A linear classifier computes a score for each class as a linear function of the input features. The class with the highest score is selected as the prediction.

For an input vector $\mathbf{x}$ and weight matrix $\mathbf{W}$:

$$\mathbf{s} = \mathbf{W} \mathbf{x}$$

where:
- $\mathbf{x} \in \mathbb{R}^D$ is the input feature vector (D dimensions)
- $\mathbf{W} \in \mathbb{R}^{C \times D}$ is the weight matrix (C classes, D features)
- $\mathbf{s} \in \mathbb{R}^C$ is the score vector for all classes

The predicted class is:

$$\hat{y} = \arg\max_j s_j$$

## SVM Loss (Hinge Loss)

### Motivation

The goal of SVM loss is to ensure that the score for the correct class is higher than the scores for incorrect classes by at least a margin $\Delta$ (typically $\Delta = 1$).

### Mathematical Formulation

For a single training example $(x_i, y_i)$ where $y_i$ is the correct class:

**Loss for one sample:**

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

where:
- $s_{y_i}$ is the score for the correct class
- $s_j$ is the score for an incorrect class $j$
- $\Delta$ is the margin (hyperparameter, typically 1)

**Interpretation:**
- If $s_{y_i} \geq s_j + \Delta$ for all $j \neq y_i$, then $L_i = 0$ (perfect classification with margin)
- Otherwise, $L_i > 0$ indicates a margin violation

### Total Loss

The total loss over all $N$ training samples is:

$$L = \frac{1}{N} \sum_{i=1}^{N} L_i$$

This is the **average hinge loss** across the dataset.

## Gradient Computation

To optimize the weights using gradient descent, we need the gradient of the loss with respect to $\mathbf{W}$.

### Gradient for a Single Sample

For sample $i$ with correct class $y_i$:

$$\frac{\partial L_i}{\partial \mathbf{W}_j} = \begin{cases}
\mathbf{x}_i & \text{if } j \neq y_i \text{ and } s_j - s_{y_i} + \Delta > 0 \\
-\sum_{j \neq y_i} \mathbb{1}(s_j - s_{y_i} + \Delta > 0) \cdot \mathbf{x}_i & \text{if } j = y_i \\
0 & \text{otherwise}
\end{cases}$$

where $\mathbb{1}(\cdot)$ is the indicator function (1 if true, 0 if false).

**Intuition:**
- For incorrect class $j$ that violates the margin: gradient pushes $\mathbf{W}_j$ away from $\mathbf{x}_i$ (reduces $s_j$)
- For correct class $y_i$: gradient pulls $\mathbf{W}_{y_i}$ toward $\mathbf{x}_i$ (increases $s_{y_i}$)
- The number of margin violations determines the magnitude of the gradient for the correct class

### Accumulated Gradient

The total gradient over all samples:

$$\nabla_{\mathbf{W}} L = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\mathbf{W}} L_i$$

## Training Algorithm

### 1. Initialization

- **Weight matrix**: $\mathbf{W} \in \mathbb{R}^{3 \times 4}$ initialized with specific values
- **Hyperparameters**:
  - Margin: $\Delta = 1$
  - Learning rate: $\eta = 0.01$
  - Convergence tolerance: $\epsilon = 10^{-4}$
  - Maximum iterations: 1000

### 2. Training Loop

For each iteration:

**a. Forward Pass (Compute Scores):**

For each training sample $\mathbf{x}_i$:

$$\mathbf{s}_i = \mathbf{W} \mathbf{x}_i$$

**b. Compute Loss:**

For each sample $i$:

$$L_i = \sum_{j \neq y_i} \max(0, s_{i,j} - s_{i,y_i} + \Delta)$$

Global loss:

$$L = \frac{1}{N} \sum_{i=1}^{N} L_i$$

**c. Compute Gradients:**

For each sample $i$, compute $\nabla_{\mathbf{W}} L_i$ as described above.

Accumulate gradients:

$$\nabla_{\mathbf{W}} L = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\mathbf{W}} L_i$$

**d. Update Weights:**

$$\mathbf{W} := \mathbf{W} - \eta \nabla_{\mathbf{W}} L$$

**e. Check Convergence:**

If $|L_{prev} - L| < \epsilon$, stop training (converged).

### 3. Testing Phase

For each test sample $\mathbf{x}_{test}$:

1. Compute scores: $\mathbf{s} = \mathbf{W} \mathbf{x}_{test}$
2. Predict class: $\hat{y} = \arg\max_j s_j$
3. Compare with true label to compute accuracy

## Code Structure

### Functions

- **`predict(xsample, W)`**: Computes score vector $\mathbf{s} = \mathbf{W} \mathbf{x}$
- **`computeLossForASample(s, labelForSample, delta)`**: Computes hinge loss $L_i$ for a single sample
- **`computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta)`**: Computes gradient $\nabla_{\mathbf{W}} L_i$ for a single sample
- **`main()`**: Orchestrates training and testing

### Dataset

**Training Data:**
- 6 samples in 4-dimensional space
- 3 classes (labeled 0, 1, 2)
- 2 samples per class

```python
x_train = [[1,5,1,4], [2,4,0,3], [2,1,3,3], [2,0,4,2], [5,1,0,2], [4,2,1,1]]
y_train = [0, 0, 1, 1, 2, 2]
```

**Test Data:**
- 3 samples (one from each class)

```python
x_test = [[1,5,2,4], [2,1,2,3], [4,1,0,1]]
y_test = [0, 1, 2]
```

## Detailed Example

### Iteration Example

Consider sample $\mathbf{x}_1 = [1, 5, 1, 4]$ with true label $y_1 = 0$.

**Step 1: Compute Scores**

$$\mathbf{s} = \mathbf{W} \mathbf{x}_1 = \begin{bmatrix} -1 & 2 & 1 & 3 \\ 2 & 0 & -1 & 4 \\ 1 & 3 & 2 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 5 \\ 1 \\ 4 \end{bmatrix} = \begin{bmatrix} 22 \\ 18 \\ 20 \end{bmatrix}$$

**Step 2: Compute Loss**

Correct class score: $s_0 = 22$

Loss contributions:
- Class 1: $\max(0, 18 - 22 + 1) = \max(0, -3) = 0$
- Class 2: $\max(0, 20 - 22 + 1) = \max(0, -1) = 0$

Total loss for this sample: $L_1 = 0$ (correctly classified with margin)

**Step 3: Compute Gradient**

Since both margin checks are satisfied ($s_j - s_{y_i} + \Delta \leq 0$), the gradient is zero for this sample.

**Step 4: Update Weights**

Weights are adjusted based on accumulated gradients from all samples.

## Convergence and Results

The algorithm iterates until:
1. The change in loss between iterations is less than the tolerance ($\epsilon = 10^{-4}$), or
2. Maximum iterations (1000) are reached

### Typical Behavior

- **Early iterations**: Loss decreases rapidly as the classifier learns to separate classes
- **Later iterations**: Loss decreases slowly as fine-tuning occurs
- **Convergence**: Loss stabilizes, indicating optimal (or near-optimal) weights

### Evaluation Metrics

**Accuracy:**

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of test samples}} \times 100\%$$

## Key Insights

### 1. Margin-Based Learning

SVM loss enforces a **safety margin** between class scores. This encourages:
- Robust classification (not just correct, but confidently correct)
- Better generalization to unseen data
- Clear decision boundaries

### 2. Subgradient Descent

The hinge loss function is **not differentiable** at points where $s_j - s_{y_i} + \Delta = 0$. However, we use **subgradient descent**, which works with any subgradient at non-differentiable points.

### 3. Linear Decision Boundaries

Each row of $\mathbf{W}$ defines a **linear decision boundary** in the feature space. For class $j$:

$$\mathbf{w}_j^T \mathbf{x} = \text{constant}$$

The classifier is limited to linearly separable problems (or approximately separable with some error).

### 4. Multi-Class Strategy

This implementation uses **one-vs-all** strategy:
- Each class has its own weight vector (row in $\mathbf{W}$)
- Score for each class is computed independently
- Highest score wins

## Comparison with Other Methods

| Aspect | SVM Loss | Softmax Loss | Perceptron |
|--------|----------|--------------|------------|
| Loss Function | Hinge loss | Cross-entropy | 0-1 loss |
| Margin | Explicit ($\Delta$) | Implicit | No margin |
| Output | Scores | Probabilities | Binary |
| Gradient | Sparse (margin violations only) | Dense (all classes) | Simple |
| Robustness | High (enforces margin) | Medium | Low |

## Advantages and Limitations

### Advantages

1. **Margin enforcement**: Robust classification with safety buffer
2. **Sparse gradients**: Only margin-violating samples contribute to updates
3. **Efficient**: Fast training for linearly separable data
4. **Interpretable**: Weight vectors represent class prototypes

### Limitations

1. **Linear only**: Cannot handle non-linearly separable data without feature engineering
2. **Sensitive to outliers**: Large margin violations can dominate the loss
3. **No probabilistic output**: Scores are not calibrated probabilities
4. **Hyperparameter tuning**: Margin $\Delta$ and learning rate require tuning

## Extensions

To improve this basic SVM classifier:

1. **Regularization**: Add $\lambda \|\mathbf{W}\|^2$ to prevent overfitting
2. **Kernel trick**: Use non-linear kernels for complex decision boundaries
3. **Soft margin**: Allow controlled margin violations with slack variables
4. **Feature scaling**: Normalize features for better convergence
5. **Mini-batch training**: Process multiple samples simultaneously for efficiency
