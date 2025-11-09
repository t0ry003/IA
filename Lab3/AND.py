import numpy as np


def activationFunction(n):
    """
    Binary step activation function.

    Args:
        n: Weighted sum input

    Returns:
        1 if n >= 0, otherwise 0
    """
    if n >= 0:
        return 1
    else:
        return 0


def forwardPropagation(p, weights, bias):
    """
    Compute forward propagation for a single perceptron.

    Args:
        p: Input vector [x1, x2]
        weights: Weight vector [w1, w2]
        bias: Scalar bias value

    Returns:
        Neuron output a in {0, 1}
    """
    # Compute weighted sum with bias
    n = np.dot(p, weights) + bias

    # Apply activation function
    a = activationFunction(n)

    return a


def main():
    """
    Train a single neuron perceptron to predict AND gate output.
    Uses the perceptron learning rule to adjust weights and bias.
    """

    # ========== Data Preparation ==========

    # Input data: All possible binary combinations
    P = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # Target labels: AND gate truth table
    t = [0, 0, 0, 1]

    # ========== Initialize Parameters ==========

    # Initialize weights with zero
    weights = [0, 0]

    # Initialize bias with zero
    bias = 0

    # Set number of training epochs
    epochs = 100

    # ========== Training Loop ==========

    for ep in range(epochs):
        for i in range(len(t)):

            # --- Forward Propagation ---
            a = forwardPropagation(P[i], weights, bias)

            # --- Compute Error ---
            error = t[i] - a

            # --- Update Weights (Perceptron Learning Rule) ---
            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]

            # --- Update Bias ---
            bias = bias + error

    # ========== Results ==========

    print("=" * 50)
    print("TRAINED PERCEPTRON PARAMETERS")
    print("=" * 50)

    print(f"\nWeights: {weights}")
    print(f"Bias: {bias}")

    print("\n" + "=" * 50)
    print("AND GATE PREDICTIONS")
    print("=" * 50 + "\n")

    for i in range(len(t)):
        a = forwardPropagation(P[i], weights, bias)
        print(f"Input: {P[i]} → Predicted: {a} | Target: {t[i]}")

    return


if __name__ == "__main__":
    main()
