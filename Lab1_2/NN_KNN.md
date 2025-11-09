# Nearest Neighbor and K-Nearest Neighbors Classification

## Overview

This code implements two fundamental **instance-based learning** algorithms for image classification:
1. **Nearest Neighbor (NN)**: Classifies based on the single closest training example
2. **K-Nearest Neighbors (KNN)**: Classifies based on majority vote of k closest training examples

Both algorithms are evaluated on the **CIFAR-10** dataset, which contains 60,000 color images (32×32 pixels) across 10 classes.

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
