# Linear Classifier with SVM Loss (Hinge Loss)

## Code Execution Walkthrough

This document explains the code execution step-by-step, following the order in which the program runs.

---

## Step 1: Import Libraries

```python
import numpy as np
```

**What happens:**
- **numpy**: Used for numerical operations, array manipulation, and matrix operations
- Essential for computing dot products, element-wise operations, and working with multi-dimensional arrays

---

## Step 2: Define `predict` Function

```python
def predict(xsample, W):
    s = np.dot(W, xsample)
    return s
```

**What happens (when called later):**

### Forward propagation - compute class scores

**Mathematical operation:**
$$\mathbf{s} = \mathbf{W} \mathbf{x}$$

**Example with actual data:**
```python
W = [[-1, 2, 1, 3],      # Weights for class 0
     [ 2, 0,-1, 4],      # Weights for class 1
     [ 1, 3, 2, 1]]      # Weights for class 2

xsample = [1, 5, 1, 4]   # Input features
```

**Computation:**
```
Score for class 0: (-1×1) + (2×5) + (1×1) + (3×4) = -1 + 10 + 1 + 12 = 22
Score for class 1: (2×1) + (0×5) + (-1×1) + (4×4) = 2 + 0 - 1 + 16 = 17
Score for class 2: (1×1) + (3×5) + (2×1) + (1×4) = 1 + 15 + 2 + 4 = 22
```

**Result:** `s = [22, 17, 22]`

**Interpretation:**
- Each class gets a score based on how well the input matches its weight vector
- Higher score = classifier thinks this class is more likely
- This is a **linear** combination of input features

---

## Step 3: Define `computeLossForASample` Function

```python
def computeLossForASample(s, labelForSample, delta):
    loss_i = 0
    syi = s[labelForSample]
    
    for idx, sj in enumerate(s):
        if idx != labelForSample:
            loss_i += max(0, sj - syi + delta)
    
    return loss_i
```

**What happens (when called later):**

### Compute SVM hinge loss for one training sample

**Purpose:** Measure how badly the classifier is doing on this sample

### 3a. Get the score for the correct class
```python
syi = s[labelForSample]
```
- Extract the score that the classifier gave to the TRUE/CORRECT class

**Example:** If true label is 0, and `s = [22, 17, 22]`, then `syi = 22`

### 3b. Compare with incorrect class scores
```python
for idx, sj in enumerate(s):
    if idx != labelForSample:
        loss_i += max(0, sj - syi + delta)
```

**What this does:**
- For each INCORRECT class, check if its score is too close to the correct class score
- `delta = 1` is the **margin** - how much better the correct class should score

**Example calculation (true label = 0):**
```
Correct class score: syi = 22

Class 1 (incorrect):
  sj - syi + delta = 17 - 22 + 1 = -4
  max(0, -4) = 0  ✓ Good! Class 1 score is much lower

Class 2 (incorrect):
  sj - syi + delta = 22 - 22 + 1 = 1
  max(0, 1) = 1   ✗ Bad! Class 2 score is too close (margin violated)

Total loss: 0 + 1 = 1
```

**Interpretation:**
- **Loss = 0**: Perfect! Correct class beats all others by at least the margin
- **Loss > 0**: Margin violation! Some incorrect class scored too high
- Goal: Make this loss as small as possible

---

## Step 4: Define `computeLossGradientForASample` Function

```python
def computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta):
    dW_i = np.zeros(W.shape)
    syi = s[labelForSample]
    
    for j, sj in enumerate(s):
        if j == labelForSample:
            continue
        
        dist = sj - syi + delta
        
        if dist > 0:
            dW_i[j] = currentDataPoint
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint
    
    return dW_i
```

**What happens (when called later):**

### Compute gradient - how to adjust weights to reduce loss

### 4a. Initialize gradient matrix
```python
dW_i = np.zeros(W.shape)
```
- Create a matrix same size as W (3×4), filled with zeros
- Will store the gradient (direction to adjust weights)

### 4b. Check each incorrect class for margin violations
```python
dist = sj - syi + delta

if dist > 0:
    dW_i[j] = currentDataPoint
    dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint
```

**What this does:**
If an incorrect class j scored too high (margin violated):

**For the incorrect class j:**
```python
dW_i[j] = currentDataPoint
```
- Gradient tells us: ADD the input to row j of W
- Effect: This will INCREASE score for class j next time
- Wait, that seems wrong? Actually, we SUBTRACT this gradient later (gradient descent)
- So we'll actually DECREASE incorrect class scores ✓

**For the correct class:**
```python
dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint
```
- Gradient tells us: SUBTRACT the input from correct class row
- After gradient descent: This will INCREASE correct class score ✓

**Example with currentDataPoint = [1, 5, 1, 4]:**

If class 2 violated the margin:
```python
dW_i[2] = [1, 5, 1, 4]           # Push class 2 down
dW_i[0] = dW_i[0] - [1, 5, 1, 4] # Pull class 0 up
      = [0, 0, 0, 0] - [1, 5, 1, 4]
      = [-1, -5, -1, -4]
```

**Result:** Returns the gradient matrix showing how to adjust weights

---

## Step 5: Main Function - Define Training Data

```python
def main():
    x_train = np.array([
        [1, 5, 1, 4],
        [2, 4, 0, 3],
        [2, 1, 3, 3],
        [2, 0, 4, 2],
        [5, 1, 0, 2],
        [4, 2, 1, 1]
    ])
    
    y_train = [0, 0, 1, 1, 2, 2]
```

**What happens:**
- **x_train**: 6 training samples, each with 4 features
  - Sample 0: [1, 5, 1, 4] belongs to class 0
  - Sample 1: [2, 4, 0, 3] belongs to class 0
  - Sample 2: [2, 1, 3, 3] belongs to class 1
  - etc.

- **y_train**: Labels (correct classes) for each sample
  - 3 classes total (0, 1, 2)
  - 2 samples per class

**Dataset visualization:**
```
Class 0: [1,5,1,4], [2,4,0,3]
Class 1: [2,1,3,3], [2,0,4,2]
Class 2: [5,1,0,2], [4,2,1,1]
```

---

## Step 6: Define Test Data

```python
x_test = np.array([
    [1, 5, 2, 4],
    [2, 1, 2, 3],
    [4, 1, 0, 1]
])

y_test = [0, 1, 2]
```

**What happens:**
- **x_test**: 3 test samples (one from each class)
- **y_test**: True labels for evaluation
- Used later to check how well the trained classifier works

---

## Step 7: Initialize Weight Matrix

```python
W = np.array([
    [-1, 2, 1, 3],
    [2, 0, -1, 4],
    [1, 3, 2, 1]
])
```

**What happens:**
- Initialize weights with specific starting values
- Shape: 3×4 (3 classes, 4 features)
- Each row represents one class's "template"
- These will be adjusted during training

**Initial weight interpretation:**
```
Class 0 template: [-1, 2, 1, 3]  means feature2 and feature4 are important
Class 1 template: [2, 0, -1, 4]  means feature1 and feature4 are important  
Class 2 template: [1, 3, 2, 1]   means feature2 and feature3 are important
```

---

## Step 8: Set Hyperparameters

```python
delta = 1            # SVM margin
step_size = 0.01     # Learning rate
tolerance = 10e-4    # Convergence threshold
max_iters = 1000     # Maximum iterations

prev_loss = 100
```

**What happens:**
- **delta (Δ = 1)**: Margin - how much better correct class must score
- **step_size (η = 0.01)**: Learning rate - how big each weight adjustment is
  - Too large: might overshoot optimal weights
  - Too small: training takes forever
- **tolerance (ε = 0.0001)**: Stop when loss stops decreasing significantly
- **max_iters**: Safety limit - stop after 1000 iterations max
- **prev_loss**: Track previous iteration's loss to detect convergence

---

## Step 9: Training Loop - Outer Iteration

```python
for iteration in range(max_iters):
    loss_L = 0
    dW = np.zeros(W.shape)
```

**What happens:**
- Start the main training loop
- Each iteration processes ALL 6 training samples
- **loss_L**: Will accumulate total loss across all samples
- **dW**: Will accumulate total gradient across all samples

---

## Step 10: Training Loop - Process Each Sample

```python
for idx, xsample in enumerate(x_train):
    s = predict(xsample, W)
    loss_i = computeLossForASample(s, y_train[idx], delta)
    dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)
    loss_L += loss_i
    dW += dW_i
```

**What happens - detailed walkthrough:**

### 10a. Forward pass - compute scores
```python
s = predict(xsample, W)
```
- Calculate scores for all 3 classes for this training sample
- Example result: `s = [22, 17, 22]`

### 10b. Compute loss
```python
loss_i = computeLossForASample(s, y_train[idx], delta)
```
- Check how badly we're doing on this sample
- If correct class doesn't beat others by margin, loss > 0

### 10c. Compute gradient
```python
dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)
```
- Calculate how to adjust weights to reduce loss for this sample

### 10d. Accumulate
```python
loss_L += loss_i
dW += dW_i
```
- Add this sample's loss to total
- Add this sample's gradient to total
- After processing all 6 samples, we'll have total loss and total gradient

**Display progress:**
```python
print(f"Sample {idx} (label={y_train[idx]}): Scores={s}, Loss={loss_i:.4f}")
```

**Example output:**
```
Sample 0 (label=0): Scores=[22 17 22], Loss=1.0000
Sample 1 (label=0): Scores=[18 15 19], Loss=2.0000
...
```

---

## Step 11: Normalize Loss and Gradient

```python
loss_L /= x_train.shape[0]
dW /= x_train.shape[0]
```

**What happens:**
- Divide by number of training samples (6)
- Get **average** loss across dataset
- Get **average** gradient across dataset

**Why average?**
- Makes the numbers more interpretable
- Makes learning rate more stable if dataset size changes

**Example:**
```
If total loss = 6.5 over 6 samples
Average loss = 6.5 / 6 = 1.083
```

---

## Step 12: Update Weights (Gradient Descent)

```python
W = W - step_size * dW
```

**What happens:**
- **Gradient descent step!**
- Adjust weights in the direction that reduces loss
- `step_size * dW` = how much to change
- **Minus sign**: Move opposite to gradient (downhill)

**Mathematical formula:**
$$\mathbf{W}_{new} = \mathbf{W}_{old} - \eta \nabla\mathbf{W} L$$

**Example:**
```python
If W[0,0] = -1.0 and dW[0,0] = 0.5
W[0,0] = -1.0 - (0.01 × 0.5) = -1.005
```

**After many iterations:** Weights converge to values that minimize loss

---

## Step 13: Check Convergence

```python
if abs(prev_loss - loss_L) < tolerance:
    print(f"CONVERGED after {iteration} iterations with loss {loss_L:.6f}")
    break

prev_loss = loss_L
```

**What happens:**
- Check if loss stopped decreasing significantly
- If `|previous_loss - current_loss| < 0.0001`, we've converged
- Stop training early (no need to continue)
- Otherwise, save current loss and continue to next iteration

**Convergence means:**
- Weights are no longer changing much
- Found (approximately) optimal weights
- Further training won't improve much

---

## Step 14: Testing Phase - Make Predictions

```python
for idx, xsample in enumerate(x_test):
    s = predict(xsample, W)
    predictedLabel = np.argmax(s)
    
    if predictedLabel == y_test[idx]:
        correctPredicted += 1
```

**What happens:**

### 14a. Compute scores with trained weights
```python
s = predict(xsample, W)
```
- Use the optimized weight matrix
- Get scores for all classes
- Example: `s = [25.3, 18.7, 22.1]`

### 14b. Predict class with highest score
```python
predictedLabel = np.argmax(s)
```
- `np.argmax` finds index of maximum value
- Example: If `s = [25.3, 18.7, 22.1]`, then `predictedLabel = 0`

### 14c. Check if correct
```python
if predictedLabel == y_test[idx]:
    correctPredicted += 1
```
- Compare prediction with true label
- Count correct predictions

---

## Step 15: Calculate Accuracy

```python
accuracy = (correctPredicted / len(x_test)) * 100

print(f"Correct predictions: {correctPredicted}/{len(x_test)}")
print(f"Test accuracy: {accuracy:.2f}%")
```

**What happens:**
- **Accuracy formula**: (Correct / Total) × 100%
- Example: If 3 out of 3 correct → 3/3 × 100 = 100%

**Output example:**
```
Correct predictions: 3/3
Test accuracy: 100.00%
```

---

## Complete Execution Flow Summary

```
1. Import numpy
2. Define predict() - computes class scores
3. Define computeLossForASample() - measures error
4. Define computeLossGradientForASample() - computes how to improve
5. Start main()
6. Load training data (6 samples, 3 classes)
7. Load test data (3 samples)
8. Initialize weight matrix W (3×4)
9. Set hyperparameters
10. FOR each iteration (up to 1000):
    a. Reset loss and gradient accumulators
    b. FOR each training sample:
       - Compute scores
       - Compute loss
       - Compute gradient
       - Accumulate
    c. Average loss and gradient
    d. Update weights using gradient descent
    e. Check convergence
    f. If converged, break
11. FOR each test sample:
    - Compute scores
    - Predict class (argmax)
    - Check if correct
12. Calculate and display accuracy
```

---

## Key Concepts in Execution Order

### Linear Classification
```
Score = W₀×feature₀ + W₁×feature₁ + W₂×feature₂ + W₃×feature₃
Prediction = class with highest score
```

### SVM Loss (Hinge Loss)
```
For each incorrect class j:
  If score_j ≥ score_correct - margin:
    Loss += (score_j - score_correct + margin)
```

### Gradient Descent
```
1. Calculate how much error (loss)
2. Calculate which direction to adjust weights (gradient)
3. Take small step in that direction
4. Repeat until error stops decreasing
```

### Time Complexity
- Each iteration: 6 samples × (score computation + loss + gradient)
- Typically converges in 10-100 iterations
- Very fast due to small dataset

---

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
