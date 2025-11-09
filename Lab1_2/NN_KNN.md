# Nearest Neighbor and K-Nearest Neighbors Classification

## Code Execution Walkthrough

This document explains the code execution step-by-step, following the order in which the program runs.

---

## Step 1: Import Libraries and Setup

```python
import numpy as np
import cv2
import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

**What happens:**
- **numpy**: Used for numerical operations and array manipulation
- **cv2**: OpenCV library (imported but not actively used in this code)
- **tensorflow**: Used to load the CIFAR-10 dataset
- **ssl configuration**: Disables SSL certificate verification to allow dataset download if there are certificate issues

---

## Step 2: Define Class Labels Dictionary

```python
DICT_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
```

**What happens:**
- Creates a mapping between numeric class labels (0-9) and human-readable class names
- CIFAR-10 has 10 classes of objects
- This dictionary is used later to display class names instead of just numbers

---

## Step 3: Define `predictLabelNN` Function

```python
def predictLabelNN(x_train_flatten, y_train, img):
    predictedLabel = -1
    scoreMin = float('inf')
    
    for idx, imgT in enumerate(x_train_flatten):
        difference = np.abs(img - imgT)
        score = np.sum(difference)
        
        if score < scoreMin:
            scoreMin = score
            predictedLabel = y_train[idx][0]
    
    return predictedLabel
```

**What happens (when called later):**

### 3a. Initialize variables
- `predictedLabel = -1`: Will hold the predicted class label
- `scoreMin = float('inf')`: Starts with infinity, will track the minimum distance found

### 3b. Loop through all training images
```python
for idx, imgT in enumerate(x_train_flatten):
```
- Iterates through all 50,000 training images
- `idx`: Current index
- `imgT`: Current training image (flattened vector)

### 3c. Compute distance
```python
difference = np.abs(img - imgT)
score = np.sum(difference)
```
- **L1 distance (Manhattan distance)**: Sum of absolute differences between pixels
- Example: If test pixel = 120 and train pixel = 100, difference = |120-100| = 20
- Sum all 3072 pixel differences to get total distance

### 3d. Track minimum distance
```python
if score < scoreMin:
    scoreMin = score
    predictedLabel = y_train[idx][0]
```
- If current training image is closer than all previous ones, update the minimum
- Store the label of this closest image
- **Result**: The function returns the label of the single most similar training image

---

## Step 4: Define `predictLabelKNN` Function

```python
def predictLabelKNN(x_train_flatten, y_train, img, k=10):
    predictions = []
    
    for idx, imgT in enumerate(x_train_flatten):
        difference = np.abs(img - imgT)
        score = np.sum(difference)
        predictions.append((score, y_train[idx][0]))
    
    predictions = sorted(predictions, key=lambda x: x[0])
    top_k_predictions = predictions[0:k]
    predLabels = [label for _, label in top_k_predictions]
    predictedLabel = max(set(predLabels), key=predLabels.count)
    
    return predictedLabel
```

**What happens (when called later):**

### 4a. Initialize list to store all distances
```python
predictions = []
```
- Will store tuples of (distance, label) for all training images

### 4b. Compute distances to ALL training images
```python
for idx, imgT in enumerate(x_train_flatten):
    difference = np.abs(img - imgT)
    score = np.sum(difference)
    predictions.append((score, y_train[idx][0]))
```
- Same distance calculation as NN
- But instead of tracking only the minimum, store ALL distances with their labels
- Example: `[(250.5, 3), (180.2, 7), (195.8, 3), ...]`

### 4c. Sort by distance
```python
predictions = sorted(predictions, key=lambda x: x[0])
```
- Sort all 50,000 (distance, label) pairs by distance (ascending)
- Smallest distances first
- Example after sorting: `[(150.1, 5), (152.3, 5), (155.7, 2), ...]`

### 4d. Select k nearest neighbors
```python
top_k_predictions = predictions[0:k]
```
- Keep only the first k entries (k=10 by default)
- These are the 10 closest training images

### 4e. Extract labels
```python
predLabels = [label for _, label in top_k_predictions]
```
- Extract just the labels from the k nearest neighbors
- Example: `[5, 5, 2, 5, 7, 5, 2, 5, 5, 2]` (10 labels)

### 4f. Majority vote
```python
predictedLabel = max(set(predLabels), key=predLabels.count)
```
- Count how many times each label appears
- Return the most common label
- In example above: label 5 appears 6 times → predict class 5
- **Result**: More robust than NN because it considers multiple neighbors

---

## Step 5: Main Function - Load Dataset

```python
def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

**What happens:**
- Downloads CIFAR-10 dataset (if not already cached)
- **x_train**: 50,000 training images, shape (50000, 32, 32, 3)
  - 50,000 images, 32×32 pixels, 3 color channels (RGB)
- **y_train**: 50,000 training labels, shape (50000, 1)
- **x_test**: 10,000 test images, shape (10000, 32, 32, 3)
- **y_test**: 10,000 test labels, shape (10000, 1)

---

## Step 6: Display Dataset Information

```python
print(f"Training images:   {x_train.shape}")
print(f"Training labels:   {y_train.shape}")
print(f"Test images:       {x_test.shape}")
print(f"Test labels:       {y_test.shape}")
```

**Output example:**
```
Training images:   (50000, 32, 32, 3)
Training labels:   (50000, 1)
Test images:       (10000, 32, 32, 3)
Test labels:       (10000, 1)
```

---

## Step 7: Display First 10 Test Labels

```python
for i in range(10):
    label_index = y_test[i][0]
    print(f"Image {i}: Label {label_index} ({DICT_CLASSES[label_index]})")
```

**What happens:**
- Shows the first 10 test images' labels
- Uses the dictionary to show both number and name

**Output example:**
```
Image 0: Label 3 (cat)
Image 1: Label 8 (ship)
Image 2: Label 8 (ship)
...
```

---

## Step 8: Flatten Images (CRUCIAL PREPROCESSING)

```python
x_train_flatten = np.float64(x_train.reshape(x_train.shape[0], 32 * 32 * 3))
x_test_flatten = np.float64(x_test.reshape(x_test.shape[0], 32 * 32 * 3))
```

**What happens:**

### Original format:
- Image: 32 × 32 × 3 = 3D tensor
- Each pixel has R, G, B values (0-255)

### After flattening:
- Image: 3072-dimensional vector (32 × 32 × 3 = 3072)
- All pixels in a single row

**Visual example:**
```
Before: [[[R,G,B], [R,G,B], ...], [[R,G,B], ...], ...]  (32×32×3)
After:  [R,G,B, R,G,B, R,G,B, ..., R,G,B]              (3072,)
```

**Why flatten?**
- Makes distance calculation simpler
- Can treat image as a point in 3072-dimensional space
- Standard format for many ML algorithms

**Data type conversion:**
- Convert to `float64` for numerical precision in calculations

**Result:**
- `x_train_flatten`: (50000, 3072)
- `x_test_flatten`: (10000, 3072)

---

## Step 9: Initialize Testing Variables

```python
numberOfCorrectPredictedImages = 0
num_images_to_test = 200
```

**What happens:**
- `numberOfCorrectPredictedImages`: Counter for correct predictions
- `num_images_to_test`: Only test on first 200 images (instead of all 10,000) to save time

**Why only 200?**
- Testing all 10,000 images would take hours
- Each test image requires comparing with 50,000 training images
- 200 images gives a reasonable accuracy estimate

---

## Step 10: Classification Loop (Main Processing)

```python
for idx, img in enumerate(x_test_flatten[0:num_images_to_test]):
```

**What happens:**
- Loops through the first 200 test images
- `idx`: Current test image index (0 to 199)
- `img`: Current flattened test image (3072-dimensional vector)

### For each test image:

#### 10a. Call prediction function
```python
predictedLabel = predictLabelNN(x_train_flatten, y_train, img)
```
- Calls `predictLabelNN` (could also use `predictLabelKNN`)
- Function compares this test image with all 50,000 training images
- Returns the predicted class label (0-9)

#### 10b. Get ground truth
```python
ground_truth_label = y_test[idx][0]
```
- Gets the actual correct label from the test set
- This is what we're trying to predict

#### 10c. Check if prediction is correct
```python
if predictedLabel == ground_truth_label:
    numberOfCorrectPredictedImages += 1
```
- If prediction matches ground truth, increment the counter
- This tracks how many images we classified correctly

#### 10d. Display result
```python
match_status = "✓" if predictedLabel == ground_truth_label else "✗"
print(f"  {match_status} Predicted: {predictedLabel} ({DICT_CLASSES[predictedLabel]}) | "
      f"True: {ground_truth_label} ({DICT_CLASSES[ground_truth_label]})")
```

**Output example:**
```
✓ Predicted: 3 (cat) | True: 3 (cat)
✗ Predicted: 5 (dog) | True: 3 (cat)
✓ Predicted: 8 (ship) | True: 8 (ship)
```

---

## Step 11: Calculate and Display Accuracy

```python
accuracy = 100 * numberOfCorrectPredictedImages / num_images_to_test

print(f"Correct predictions: {numberOfCorrectPredictedImages}/{num_images_to_test}")
print(f"Accuracy: {accuracy:.2f}%")
```

**What happens:**
- **Accuracy formula**: (Correct predictions / Total predictions) × 100%
- Example: If 72 out of 200 were correct → 72/200 × 100 = 36%

**Output example:**
```
Correct predictions: 72/200
Accuracy: 36.00%
```

**Interpretation:**
- Random guessing would be 10% (10 classes)
- NN typically achieves 35-38% on CIFAR-10
- KNN slightly better at 38-40%
- Modern CNNs achieve >95%

---

## Complete Execution Flow Summary

```
1. Import libraries → Setup SSL
2. Define class labels dictionary
3. Define predictLabelNN function (not executed yet)
4. Define predictLabelKNN function (not executed yet)
5. Start main()
6. Load CIFAR-10 dataset (50,000 train + 10,000 test images)
7. Display dataset shapes
8. Show first 10 test labels
9. Flatten all images: 32×32×3 → 3072-dimensional vectors
10. Initialize counters
11. For each of 200 test images:
    a. Compare with all 50,000 training images
    b. Find nearest neighbor(s)
    c. Predict label
    d. Check if correct
    e. Display result
12. Calculate final accuracy
13. Display results
```

---

## Key Concepts in Execution Order

### Distance Calculation (L1)
For each pair of images:
```
Distance = |pixel₁ - pixel₁'| + |pixel₂ - pixel₂'| + ... + |pixel₃₀₇₂ - pixel₃₀₇₂'|
```

### NN Decision
```
Predicted class = Label of training image with minimum distance
```

### KNN Decision  
```
1. Find k closest training images
2. Count votes for each class
3. Predicted class = Most frequent class among k neighbors
```

### Time Complexity
- For 1 test image: 50,000 × 3072 operations ≈ 153 million operations
- For 200 test images: 30.7 billion operations
- This is why the code takes several minutes to run!

---

## The CIFAR-10 Dataset

### Dataset Composition

**Classes (10 total):**
- 0: Airplane
- 1: Automobile
- 2: Bird
- 3: Cat
- 4: Deer
- 5: Dog
- 6: Frog
- 7: Horse
- 8: Ship
- 9: Truck

**Structure:**
- **Training set**: 50,000 images (5,000 per class)
- **Test set**: 10,000 images (1,000 per class)
- **Image dimensions**: 32 × 32 × 3 (width × height × RGB channels)

## Nearest Neighbor (NN) Classifier

### Algorithm Description

The Nearest Neighbor classifier uses a simple principle: **classify a test image based on the most similar training image**.

### Mathematical Formulation

Given:
- Test image: $\mathbf{x}_{test} \in \mathbb{R}^D$
- Training set: $\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_N, y_N)\}$

**Step 1: Compute distances**

For each training image $\mathbf{x}_i$, compute the distance to the test image:

$$d_i = d(\mathbf{x}_{test}, \mathbf{x}_i)$$

**Step 2: Find nearest neighbor**

$$i^* = \arg\min_{i} d_i$$

**Step 3: Predict label**

$$\hat{y} = y_{i^*}$$

### Distance Metric: L1 (Manhattan Distance)

This implementation uses **L1 distance** (also called Manhattan distance or City Block distance):

$$d_{L1}(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{D} |x_j - x'_j|$$

where:
- $D$ is the dimensionality (3072 for CIFAR-10 flattened images)
- $x_j$ is the $j$-th pixel value in image $\mathbf{x}$

**Alternative: L2 Distance (Euclidean)**

Another common metric is L2 distance:

$$d_{L2}(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{D} (x_j - x'_j)^2}$$

### Algorithm Complexity

- **Training time**: $O(1)$ (just store the data)
- **Prediction time**: $O(N \cdot D)$ where $N$ = training samples, $D$ = dimensions
- **Space complexity**: $O(N \cdot D)$ (store all training data)

### Characteristics

**Advantages:**
- Simple to implement
- No training phase (lazy learning)
- Works with any number of classes
- No assumptions about data distribution

**Disadvantages:**
- Very slow at test time (must compare with all training samples)
- Sensitive to noise and outliers (single nearest neighbor)
- Curse of dimensionality (performance degrades in high dimensions)
- No learned decision boundary

## K-Nearest Neighbors (KNN) Classifier

### Algorithm Description

KNN improves upon NN by considering **multiple neighbors** and using **majority voting** to make predictions. This makes it more robust to noise.

### Mathematical Formulation

Given:
- Test image: $\mathbf{x}_{test}$
- Parameter: $k$ (number of neighbors)

**Step 1: Compute all distances**

$$d_i = d(\mathbf{x}_{test}, \mathbf{x}_i) \quad \text{for all } i = 1, ..., N$$

**Step 2: Find k nearest neighbors**

Sort distances and select indices of k smallest:

$$\mathcal{N}_k = \{i_1, i_2, ..., i_k\} \quad \text{where } d_{i_1} \leq d_{i_2} \leq ... \leq d_{i_k}$$

**Step 3: Majority vote**

$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k} \mathbb{1}(y_i = c)$$

where $\mathbb{1}(\cdot)$ is the indicator function (1 if true, 0 if false).

In other words, predict the class that appears most frequently among the k nearest neighbors.

### Choosing k

The parameter $k$ controls the bias-variance tradeoff:

- **Small k (e.g., k=1)**: 
  - Low bias, high variance
  - Sensitive to noise
  - Complex decision boundaries
  
- **Large k (e.g., k=100)**:
  - High bias, low variance
  - More robust to noise
  - Smoother decision boundaries
  - Risk of including irrelevant neighbors

**Typical values**: $k \in \{1, 3, 5, 10, 20, 50\}$

**Rule of thumb**: Try $k = \sqrt{N}$ as a starting point.

In this implementation, $k = 10$ is used.

### Algorithm Complexity

- **Training time**: $O(1)$ (lazy learning)
- **Prediction time**: $O(N \cdot D + N \log k)$ 
  - $O(N \cdot D)$ to compute distances
  - $O(N \log k)$ to find k smallest (using heap)
  - Can be optimized with approximate nearest neighbor algorithms
- **Space complexity**: $O(N \cdot D)$

### Characteristics

**Advantages over NN:**
- More robust to noise (majority vote)
- Better decision boundaries
- Tunable complexity via $k$

**Disadvantages:**
- Still slow at test time
- Requires choosing appropriate $k$
- Class imbalance can affect voting

## Implementation Details

### Image Preprocessing

**Original format**: 32 × 32 × 3 (3D tensor)

**Flattened format**: 3072-dimensional vector

$$\mathbf{x}_{flattened} = [R_{0,0}, G_{0,0}, B_{0,0}, R_{0,1}, ..., B_{31,31}]$$

**Why flatten?**
- Simplifies distance computation
- Standard format for many ML algorithms
- Each pixel becomes a feature

**Data type conversion**: Convert to `float64` for numerical stability

### Distance Computation

For each test image, compute distance to all 50,000 training images:

```python
difference = np.abs(img - imgT)  # Element-wise absolute difference
score = np.sum(difference)        # Sum all pixel differences
```

This computes:

$$d_{L1} = \sum_{j=1}^{3072} |x_{test,j} - x_{train,j}|$$

### Memory and Computational Considerations

**Memory requirements:**
- Training data: 50,000 × 3072 × 8 bytes ≈ 1.2 GB
- Test data: 10,000 × 3072 × 8 bytes ≈ 0.24 GB

**Computational cost for 200 test images:**
- NN: 200 × 50,000 × 3072 operations ≈ 30 billion operations
- KNN: Similar + sorting overhead

**Optimization possibilities:**
1. Use vectorized operations (NumPy broadcasting)
2. Approximate nearest neighbor algorithms (e.g., KD-trees, LSH)
3. Dimensionality reduction (PCA)
4. GPU acceleration

## Code Structure

### Functions

**`predictLabelNN(x_train_flatten, y_train, img)`**
- Implements Nearest Neighbor classification
- Returns the label of the closest training image

**`predictLabelKNN(x_train_flatten, y_train, img, k=10)`**
- Implements K-Nearest Neighbors classification
- Returns the majority class among k nearest neighbors

**`main()`**
- Loads CIFAR-10 dataset
- Preprocesses images (flattening)
- Evaluates classifier on 200 test images
- Reports accuracy

### Evaluation Process

1. **Load dataset** using TensorFlow/Keras
2. **Explore data**: Print shapes and first 10 labels
3. **Preprocess**: Flatten images to vectors
4. **Classify**: For each of 200 test images:
   - Compute distances to all training images
   - Find nearest neighbor(s)
   - Predict label
   - Compare with ground truth
5. **Evaluate**: Calculate accuracy

### Accuracy Metric

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \times 100\%$$

## Expected Performance

### Typical Results on CIFAR-10

**Nearest Neighbor (NN):**
- Accuracy: ~35-38%
- Random guessing: 10% (10 classes)
- Better than random, but not great

**K-Nearest Neighbors (KNN, k=10):**
- Accuracy: ~38-40%
- Slight improvement over NN
- More stable predictions

**Why is performance limited?**
1. **Pixel-based comparison**: Images that look different visually can be similar pixel-wise
2. **No feature learning**: Raw pixels are not ideal features
3. **High dimensionality**: 3072 dimensions with limited training data
4. **Image variations**: Same object can look very different (lighting, angle, etc.)

### Comparison with Other Methods

| Method | CIFAR-10 Accuracy | Complexity |
|--------|-------------------|------------|
| Random | 10% | O(1) |
| NN (L1) | ~35% | O(N·D) |
| KNN (k=10) | ~38% | O(N·D) |
| Linear SVM | ~40% | O(D) after training |
| CNN (modern) | >95% | O(D) after training |

## Visualizing Decision Boundaries

### 1D Example

Consider a simple 1D case with 2 classes (red and blue points):

**NN Decision Boundary:**
- Creates Voronoi cells around each point
- Each cell assigns its training point's label
- Very irregular, can overfit

**KNN Decision Boundary (k=3):**
- Smoother boundaries
- Less sensitive to individual outliers
- More generalized

### High-Dimensional Challenge

In high dimensions (like 3072 for CIFAR-10):
- **Curse of dimensionality**: All points become approximately equidistant
- **Sparse data**: Training samples don't densely cover the space
- **Solution**: Feature learning or dimensionality reduction

## Improvements and Extensions

### 1. Distance Weighting

Weight closer neighbors more heavily:

$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k} w_i \cdot \mathbb{1}(y_i = c)$$

where $w_i = \frac{1}{d_i + \epsilon}$ (inverse distance weighting)

### 2. Better Distance Metrics

- **Chi-squared distance** for histograms
- **Cosine similarity** for high-dimensional data
- **Learned metrics** (metric learning)

### 3. Dimensionality Reduction

- **PCA** (Principal Component Analysis)
- **Autoencoders**
- **Feature extraction** (HOG, SIFT)

### 4. Data Augmentation

Increase training data with:
- Rotations
- Translations
- Color jittering
- Horizontal flips

### 5. Efficient Nearest Neighbor Search

- **KD-trees**: For low-dimensional data (D < 20)
- **Ball trees**: Better for high dimensions
- **Locality-Sensitive Hashing (LSH)**: Approximate but fast
- **FAISS** (Facebook AI Similarity Search): GPU-accelerated

### 6. Deep Learning Features

Instead of raw pixels, use features from a pre-trained CNN:
```
Image → CNN (e.g., ResNet) → Feature Vector → KNN
```

This typically achieves much better accuracy.

## Key Takeaways

1. **NN and KNN are simple but powerful** baseline methods
2. **Instance-based learning** requires no training but is slow at test time
3. **KNN generalizes better than NN** through majority voting
4. **Raw pixel distances** are suboptimal for image classification
5. **Modern deep learning** methods significantly outperform NN/KNN on CIFAR-10
6. **These algorithms are still useful** for:
   - Quick prototyping
   - Small datasets
   - Non-parametric problems
   - Anomaly detection
   - With good feature representations

## Conclusion

While Nearest Neighbor and K-Nearest Neighbors classifiers achieve modest performance on CIFAR-10 (~35-40% accuracy), they serve as important baseline methods and illustrate fundamental concepts in machine learning:

- **Distance-based classification**
- **Instance-based learning** (lazy learning)
- **Bias-variance tradeoff** (through k)
- **Importance of feature representation**

For production systems on complex image datasets, deep learning approaches (CNNs) are preferred, but NN/KNN remain valuable tools for understanding, prototyping, and specific use cases.
