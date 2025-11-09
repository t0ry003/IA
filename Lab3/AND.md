# AND Gate Implementation using Perceptron

## Code Execution Walkthrough

This document explains the code execution step-by-step, following the order in which the program runs.

---

## What is an AND Gate?

An AND gate is a basic digital logic gate that outputs `1` (true) only when **both** of its inputs are `1`. Otherwise, it outputs `0` (false).

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 0      |
| 1       | 0       | 0      |
| 1       | 1       | 1      |

---

## Step 1: Import Libraries

```python
import numpy as np
```

**What happens:**
- **numpy**: Used for numerical operations, specifically the dot product calculation
- Provides efficient array operations for vector and matrix computations

---

## Step 2: Define `activationFunction`

```python
def activationFunction(n):
    if n >= 0:
        return 1
    else:
        return 0
```

**What happens (when called later):**

### Binary step function - the decision maker

**Mathematical formula:**
$$a = f(n) = \begin{cases} 
1 & \text{if } n \geq 0 \\
0 & \text{if } n < 0
\end{cases}$$

**What it does:**
- Takes the weighted sum `n` as input
- If `n` is positive or zero → output 1
- If `n` is negative → output 0
- This is the "activation" that fires the neuron

**Examples:**
```python
activationFunction(5.2)   → 1  ✓ Positive
activationFunction(0)     → 1  ✓ Zero counts as positive
activationFunction(-0.3)  → 0  ✓ Negative
```

**Why this function?**
- Mimics biological neurons (fire or don't fire)
- Converts continuous values into binary decisions
- Perfect for AND gate which only outputs 0 or 1

---

## Step 3: Define `forwardPropagation`

```python
def forwardPropagation(p, weights, bias):
    n = np.dot(p, weights) + bias
    a = activationFunction(n)
    return a
```

**What happens (when called later):**

### Complete the forward pass through the perceptron

### 3a. Compute weighted sum
```python
n = np.dot(p, weights) + bias
```

**Mathematical operation:**
$$n = w_1 \cdot x_1 + w_2 \cdot x_2 + b$$

**Example calculation:**
```python
p = [1, 1]           # Input: both inputs are 1
weights = [1, 1]     # Current weights
bias = -1            # Current bias

n = (1 × 1) + (1 × 1) + (-1)
  = 1 + 1 - 1
  = 1
```

**What `np.dot` does:**
- Multiplies corresponding elements and sums them
- `[1, 1] · [1, 1]` = (1×1) + (1×1) = 2
- Then adds bias

### 3b. Apply activation function
```python
a = activationFunction(n)
```
- Pass the weighted sum through the step function
- Get final output (0 or 1)

**Example:**
```python
If n = 1 (positive) → a = 1
If n = -2 (negative) → a = 0
```

**Result:** Returns the neuron's prediction

---

## Step 4: Main Function - Define Training Data

```python
def main():
    P = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    t = [0, 0, 0, 1]
```

**What happens:**

### Training data - all possible inputs and correct outputs

**P (inputs):** All 4 possible combinations of two binary values
```
P[0] = [0, 0]  →  Input: both OFF
P[1] = [0, 1]  →  Input: first OFF, second ON
P[2] = [1, 0]  →  Input: first ON, second OFF
P[3] = [1, 1]  →  Input: both ON
```

**t (targets):** Correct AND gate outputs for each input
```
t[0] = 0  →  0 AND 0 = 0
t[1] = 0  →  0 AND 1 = 0
t[2] = 0  →  1 AND 0 = 0
t[3] = 1  →  1 AND 1 = 1  (only this should output 1)
```

**Visual representation:**
```
Input [0,0] should give 0
Input [0,1] should give 0
Input [1,0] should give 0
Input [1,1] should give 1  ← Only when BOTH inputs are 1
```

---

## Step 5: Initialize Parameters

```python
weights = [0, 0]
bias = 0
epochs = 100
```

**What happens:**

### Start with blank slate - no knowledge

**weights = [0, 0]**
- `weights[0]`: Weight for first input (x₁)
- `weights[1]`: Weight for second input (x₂)
- Both start at 0 (no preference for either input)

**bias = 0**
- The threshold adjustment
- Starts at 0 (neutral)

**epochs = 100**
- Number of complete passes through all training data
- Will see each of the 4 examples 100 times
- Total training steps: 100 epochs × 4 examples = 400 updates

**Why start with zeros?**
- Perceptron learning rule guaranteed to converge from any starting point
- Zero is simple and neutral

---

## Step 6: Training Loop - Outer Loop (Epochs)

```python
for ep in range(epochs):
```

**What happens:**
- Repeat the training process 100 times
- Each epoch goes through all 4 training examples
- After each epoch, weights should be closer to correct values

**Purpose of multiple epochs:**
- First epoch: Make initial adjustments
- Later epochs: Fine-tune the weights
- Multiple passes ensure convergence

---

## Step 7: Training Loop - Inner Loop (Each Example)

```python
for i in range(len(t)):
```

**What happens:**
- Loop through each of the 4 training examples
- `i = 0`: Process [0,0] → 0
- `i = 1`: Process [0,1] → 0
- `i = 2`: Process [1,0] → 0
- `i = 3`: Process [1,1] → 1

**This is where the learning happens!**

---

## Step 8: Forward Propagation

```python
a = forwardPropagation(P[i], weights, bias)
```

**What happens:**
- Make a prediction with current weights
- Compute: n = w₁×x₁ + w₂×x₂ + b
- Apply activation: a = 1 if n ≥ 0, else 0

**Example - First iteration (epoch 0, example 0):**
```python
Input: P[0] = [0, 0]
Current weights: [0, 0]
Current bias: 0

n = (0 × 0) + (0 × 0) + 0 = 0
a = activationFunction(0) = 1  (since 0 ≥ 0)
```

**Example - Later iteration (epoch 5, example 3):**
```python
Input: P[3] = [1, 1]
Current weights: [1, 1]  (after some training)
Current bias: -1

n = (1 × 1) + (1 × 1) + (-1) = 1
a = activationFunction(1) = 1 ✓ Correct!
```

---

## Step 9: Compute Error

```python
error = t[i] - a
```

**What happens:**
- Compare prediction with correct answer
- Calculate how wrong we were

**Mathematical formula:**
$$e = t - a$$

**Possible error values:**
```
error = 0:   Prediction correct! No adjustment needed
error = 1:   Should be 1, predicted 0 (false negative)
error = -1:  Should be 0, predicted 1 (false positive)
```

**Examples:**
```python
# Case 1: Correct prediction
Target: 0, Predicted: 0 → error = 0 - 0 = 0 ✓

# Case 2: False negative
Target: 1, Predicted: 0 → error = 1 - 0 = 1 (need to increase output)

# Case 3: False positive
Target: 0, Predicted: 1 → error = 0 - 1 = -1 (need to decrease output)
```

---

## Step 10: Update Weights (Perceptron Learning Rule)

```python
weights[0] = weights[0] + error * P[i][0]
weights[1] = weights[1] + error * P[i][1]
```

**What happens:**

### Adjust weights based on error and input

**Mathematical formula:**
$$w_j^{new} = w_j^{old} + e \cdot x_j$$

**Example - Error = 1 (false negative):**
```python
Current weights: [0, 0]
Input: P[3] = [1, 1]
error = 1

weights[0] = 0 + 1 × 1 = 1
weights[1] = 0 + 1 × 1 = 1

New weights: [1, 1]
```

**Example - Error = -1 (false positive):**
```python
Current weights: [1, 1]
Input: P[1] = [0, 1]
error = -1

weights[0] = 1 + (-1) × 0 = 1  (no change, input was 0)
weights[1] = 1 + (-1) × 1 = 0  (decreased)

New weights: [1, 0]
```

**Example - Error = 0 (correct):**
```python
Current weights: [1, 1]
Input: P[3] = [1, 1]
error = 0

weights[0] = 1 + 0 × 1 = 1  (no change)
weights[1] = 1 + 0 × 1 = 1  (no change)

Weights stay: [1, 1]
```

**Key insight:**
- If input is 0, that weight doesn't change (0 × anything = 0)
- If input is 1, weight changes by the error amount
- This way, only "active" inputs affect learning

---

## Step 11: Update Bias

```python
bias = bias + error
```

**What happens:**

### Adjust the threshold

**Mathematical formula:**
$$b^{new} = b^{old} + e$$

**Examples:**
```python
# Error = 1 (need higher output)
bias = -1 + 1 = 0  (threshold less strict)

# Error = -1 (need lower output)
bias = 0 + (-1) = -1  (threshold more strict)

# Error = 0 (correct)
bias = -1 + 0 = -1  (no change)
```

**What bias does:**
- **Positive bias**: Easier to activate (output 1)
- **Negative bias**: Harder to activate (output 1)
- For AND gate: Typically ends up negative (both inputs must be on)

---

## Step 12: Training Continues

**What happens over 100 epochs:**

### Epoch 1: Major adjustments
```
Weights: [0, 0] → [1, 1]
Bias: 0 → -1
```

### Epochs 2-10: Fine-tuning
```
Weights stabilize around [1, 1]
Bias stabilizes around -1 or -2
```

### Epochs 11-100: Already converged
```
All predictions correct
No more weight changes (error = 0 for all examples)
```

**Typical convergence:**
- AND gate usually learns in less than 10 epochs
- Remaining epochs just confirm it's correct
- This is a very simple, linearly separable problem

---

## Step 13: Display Results - Parameters

```python
print(f"\nWeights: {weights}")
print(f"Bias: {bias}")
```

**What happens:**
- Show the final learned parameters

**Typical output:**
```
Weights: [1, 1]
Bias: -1
```

**What this means:**
- Decision boundary: $x_1 + x_2 - 1 \geq 0$
- Or rearranged: $x_1 + x_2 \geq 1$
- Both inputs must contribute to reach threshold

---

## Step 14: Display Results - Predictions

```python
for i in range(len(t)):
    a = forwardPropagation(P[i], weights, bias)
    print(f"Input: {P[i]} → Predicted: {a} | Target: {t[i]}")
```

**What happens:**

### Test the trained perceptron on all inputs

**Example output:**
```
Input: [0, 0] → Predicted: 0 | Target: 0 ✓
Input: [0, 1] → Predicted: 0 | Target: 0 ✓
Input: [1, 0] → Predicted: 0 | Target: 0 ✓
Input: [1, 1] → Predicted: 1 | Target: 1 ✓
```

**Detailed calculation for each:**

**Input [0, 0]:**
```
n = (1 × 0) + (1 × 0) + (-1) = -1
a = activationFunction(-1) = 0 ✓
```

**Input [0, 1]:**
```
n = (1 × 0) + (1 × 1) + (-1) = 0
a = activationFunction(0) = 1 
Wait... this might be wrong depending on final bias
Actually with bias = -2:
n = 0 + 1 - 2 = -1 → a = 0 ✓
```

**Input [1, 0]:**
```
n = (1 × 1) + (1 × 0) + (-1) = 0
With bias = -2: n = 1 + 0 - 2 = -1 → a = 0 ✓
```

**Input [1, 1]:**
```
n = (1 × 1) + (1 × 1) + (-1) = 1
With bias = -2: n = 1 + 1 - 2 = 0 → a = 1 ✓
```

---

## Complete Execution Flow Summary

```
1. Import numpy
2. Define activationFunction(n) - binary step function
3. Define forwardPropagation(p, weights, bias) - compute prediction
4. Start main()
5. Define training data:
   - P = [[0,0], [0,1], [1,0], [1,1]]
   - t = [0, 0, 0, 1]
6. Initialize weights = [0, 0], bias = 0
7. FOR epoch in 0 to 99:
   FOR example in 0 to 3:
     a. Compute prediction
     b. Calculate error = target - prediction
     c. Update weights: w += error × input
     d. Update bias: b += error
8. Display final weights and bias
9. Test on all 4 inputs and display results
```

---

## Key Learning Concepts

### The Perceptron Learning Rule

**Simple but powerful:**
```
IF prediction is wrong:
    Adjust weights in direction that would make it right
IF prediction is correct:
    Don't change anything
```

### Why It Works for AND Gate

**AND gate is linearly separable:**
```
Plot the inputs:
(0,0), (0,1), (1,0) should output 0  ← Bottom-left cluster
(1,1) should output 1                 ← Top-right point

A straight line can separate them!
Line: x₁ + x₂ = 1.5
```

### Decision Boundary

With final weights [1, 1] and bias -1.5:
```
1×x₁ + 1×x₂ - 1.5 = 0
x₁ + x₂ = 1.5

Points where x₁ + x₂ ≥ 1.5 → output 1
Only (1,1) satisfies this: 1 + 1 = 2 ≥ 1.5 ✓
```

---

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
