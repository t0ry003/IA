import numpy as np


def predict(xsample, W):
    """
    Compute scores for all classes given an input sample.

    Args:
        xsample: Input feature vector
        W: Weight matrix (classes × features)

    Returns:
        Score vector for all classes
    """
    s = np.dot(W, xsample)
    return s


def computeLossForASample(s, labelForSample, delta):
    """
    Compute the SVM hinge loss for a single data point.

    Args:
        s: Score vector for all classes
        labelForSample: True label index for the sample
        delta: Margin parameter (typically 1)

    Returns:
        Loss value for this sample
    """
    loss_i = 0
    # Score for the correct class
    syi = s[labelForSample]

    # Compute hinge loss: max(0, sj - syi + delta) for all j != yi
    for idx, sj in enumerate(s):
        if idx != labelForSample:
            loss_i += max(0, sj - syi + delta)

    return loss_i


def computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta):
    """
    Compute the gradient of the loss with respect to weights for a single sample.

    Args:
        W: Weight matrix
        s: Score vector for all classes
        currentDataPoint: Input feature vector
        labelForSample: True label index
        delta: Margin parameter

    Returns:
        Gradient matrix dW_i for this sample
    """
    # Initialize gradient matrix with zeros
    dW_i = np.zeros(W.shape)

    # Score for the correct class
    syi = s[labelForSample]

    for j, sj in enumerate(s):
        # Skip the correct class
        if j == labelForSample:
            continue

        # Compute margin
        dist = sj - syi + delta

        # If margin violation occurs
        if dist > 0:
            # Gradient for incorrect class j
            dW_i[j] = currentDataPoint
            # Gradient for correct class (accumulated negative)
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint

    return dW_i


def main():
    """
    Train a linear classifier using SVM loss (hinge loss).
    Uses gradient descent to optimize the weight matrix.
    """

    # ========== Data Preparation ==========

    # Training data: 6 samples in 4-dimensional space
    x_train = np.array([
        [1, 5, 1, 4],
        [2, 4, 0, 3],
        [2, 1, 3, 3],
        [2, 0, 4, 2],
        [5, 1, 0, 2],
        [4, 2, 1, 1]
    ])

    # Training labels: 3 classes (0, 1, 2)
    y_train = [0, 0, 1, 1, 2, 2]

    # Test data: 3 samples
    x_test = np.array([
        [1, 5, 2, 4],
        [2, 1, 2, 3],
        [4, 1, 0, 1]
    ])

    # Test labels
    y_test = [0, 1, 2]

    # ========== Initialize Parameters ==========

    # Weight matrix: 3 classes × 4 features
    W = np.array([
        [-1, 2, 1, 3],
        [2, 0, -1, 4],
        [1, 3, 2, 1]
    ])

    # Hyperparameters
    delta = 1            # SVM margin
    step_size = 0.01     # Learning rate
    tolerance = 10e-4    # Convergence threshold
    max_iters = 1000     # Maximum iterations

    prev_loss = 100

    # ========== Training Loop ==========

    print("=" * 60)
    print("TRAINING LINEAR CLASSIFIER WITH SVM LOSS")
    print("=" * 60 + "\n")

    for iteration in range(max_iters):
        loss_L = 0
        dW = np.zeros(W.shape)

        # --- Process Each Training Sample ---
        for idx, xsample in enumerate(x_train):

            # Compute scores for all classes
            s = predict(xsample, W)

            # Compute loss for this sample
            loss_i = computeLossForASample(s, y_train[idx], delta)

            # Display scores and loss
            print(
                f"Sample {idx} (label={y_train[idx]}): Scores={s}, Loss={loss_i:.4f}")

            # Compute gradient for this sample
            dW_i = computeLossGradientForASample(
                W, s, x_train[idx], y_train[idx], delta)

            # Accumulate global loss
            loss_L += loss_i

            # Accumulate global gradient
            dW += dW_i

        # --- Normalize Loss and Gradient ---

        # Average loss over all samples
        loss_L /= x_train.shape[0]
        print(
            f"\nIteration {iteration}: Global normalized loss = {loss_L:.6f}\n")

        # Average gradient over all samples
        dW /= x_train.shape[0]

        # --- Update Weights ---

        W = W - step_size * dW

        # --- Check Convergence ---

        if abs(prev_loss - loss_L) < tolerance:
            print("=" * 60)
            print(
                f"CONVERGED after {iteration} iterations with loss {loss_L:.6f}")
            print("=" * 60 + "\n")
            break

        prev_loss = loss_L

    # ========== Testing Phase ==========

    print("=" * 60)
    print("TESTING PHASE")
    print("=" * 60 + "\n")

    correctPredicted = 0

    for idx, xsample in enumerate(x_test):
        # Compute scores
        s = predict(xsample, W)

        # Predict class with highest score
        predictedLabel = np.argmax(s)

        # Check if prediction is correct
        if predictedLabel == y_test[idx]:
            correctPredicted += 1

        print(
            f"Test sample {idx}: Predicted={predictedLabel}, True={y_test[idx]}, Scores={s}")

    # ========== Results ==========

    accuracy = (correctPredicted / len(x_test)) * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Correct predictions: {correctPredicted}/{len(x_test)}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print("=" * 60)

    return


if __name__ == '__main__':
    main()
