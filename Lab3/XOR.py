import numpy as np


def sigmoid(n):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-n))


def sigmoidDerivative(n):
    """Derivative of the sigmoid function."""
    return n * (1 - n)


def forwardPropagationLayer(p, weights, biases):
    """
    Compute forward propagation for a single layer.

    Args:
        p: Input vector
        weights: Weight matrix
        biases: Bias vector

    Returns:
        Layer output after activation
    """
    # Compute weighted sum with bias
    n = np.dot(p, weights) + biases

    # Apply activation function
    a = sigmoid(n)

    return a


def main():
    """
    Train an Artificial Neural Network to predict XOR gate output.
    The network uses backpropagation with two layers (hidden + output).
    """

    # ========== Data Preparation ==========

    # Input data: All possible binary combinations
    points = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Target labels: XOR truth table
    labels = np.array([[0], [1], [1], [0]])

    # ========== Network Architecture ==========

    inputSize = 2
    noNeuronsLayer1 = 2  # Hidden layer
    noNeuronsLayer2 = 1  # Output layer

    # Initialize weights and biases with random values
    weightsLayer1 = np.random.uniform(size=(inputSize, noNeuronsLayer1))
    weightsLayer2 = np.random.uniform(size=(noNeuronsLayer1, noNeuronsLayer2))

    biasLayer1 = np.random.uniform(size=(1, noNeuronsLayer1))
    biasLayer2 = np.random.uniform(size=(1, noNeuronsLayer2))

    # ========== Training Parameters ==========

    noEpochs = 5000
    learningRate = 0.3

    # ========== Training Loop ==========

    for epoch in range(noEpochs):

        # --- Forward Propagation ---
        hidden_layer_output = forwardPropagationLayer(
            points, weightsLayer1, biasLayer1
        )
        predicted_output = forwardPropagationLayer(
            hidden_layer_output, weightsLayer2, biasLayer2
        )

        # --- Backpropagation ---

        # Compute error and gradient for output layer
        bkProp_error = labels - predicted_output
        d_predicted_output = bkProp_error * sigmoidDerivative(predicted_output)

        # Compute error and gradient for hidden layer
        error_hidden_layer = d_predicted_output.dot(weightsLayer2.T)
        d_hidden_layer = error_hidden_layer * \
            sigmoidDerivative(hidden_layer_output)

        # --- Update Weights and Biases ---

        # Update layer 2 (output layer)
        weightsLayer2 += hidden_layer_output.T.dot(
            d_predicted_output) * learningRate
        biasLayer2 += np.sum(d_predicted_output, axis=0,
                             keepdims=True) * learningRate

        # Update layer 1 (hidden layer)
        weightsLayer1 += points.T.dot(d_hidden_layer) * learningRate
        biasLayer1 += np.sum(d_hidden_layer, axis=0,
                             keepdims=True) * learningRate

        # --- Compute Cost ---

        C = np.mean((labels - predicted_output) ** 2) / 2

        # Display progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Error = {C:.6f}")

        # Early stopping if error is small enough
        if C < 0.01:
            print(f"\nTraining stopped at epoch {epoch} with error = {C:.6f}")
            break

    # ========== Results ==========

    print("\n" + "=" * 50)
    print("TRAINED NETWORK PARAMETERS")
    print("=" * 50)

    print("\nLayer 1 (Hidden Layer):")
    print(f"  Weights:\n{weightsLayer1}")
    print(f"  Biases:\n{biasLayer1}")

    print("\nLayer 2 (Output Layer):")
    print(f"  Weights:\n{weightsLayer2}")
    print(f"  Biases:\n{biasLayer2}")

    print("\n" + "=" * 50)
    print("XOR GATE PREDICTIONS")
    print("=" * 50)

    for i in range(len(labels)):
        outL1 = forwardPropagationLayer(points[i], weightsLayer1, biasLayer1)
        outL2 = forwardPropagationLayer(outL1, weightsLayer2, biasLayer2)

        print(
            f"Input: {points[i]} → Predicted: {float(outL2[0]):.4f} | Target: {labels[i][0]}")


if __name__ == "__main__":
    main()
