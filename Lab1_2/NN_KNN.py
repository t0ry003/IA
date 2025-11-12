import numpy as np
import cv2
import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR-10 class labels
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


def predictLabelNN(x_train_flatten, y_train, img):
    """
    Nearest Neighbor (NN) classifier - predicts label based on the single closest training image.

    Uses L1 distance (Manhattan distance) to find the most similar training image.

    Args:
        x_train_flatten: Flattened training images (N × D)
        y_train: Training labels (N × 1)
        img: Test image to classify (flattened vector)

    Returns:
        Predicted label (class index)
    """
    predictedLabel = -1
    scoreMin = float('inf')

    # Compare test image with every training image
    for idx, imgT in enumerate(x_train_flatten):

        # Compute L1 (Manhattan) distance: sum of absolute differences
        difference = np.abs(img - imgT)
        score = np.sum(difference)

        # Track the training image with minimum distance
        if score < scoreMin:
            scoreMin = score
            predictedLabel = y_train[idx][0]

    return predictedLabel


def predictLabelKNN(x_train_flatten, y_train, img, k=10):
    """
    K-Nearest Neighbors (KNN) classifier - predicts label based on majority vote of k closest images.

    Args:
        x_train_flatten: Flattened training images (N × D)
        y_train: Training labels (N × 1)
        img: Test image to classify (flattened vector)
        k: Number of nearest neighbors to consider (default: 10)

    Returns:
        Predicted label (class index)
    """
    predictions = []

    # Compute distances to all training images
    for idx, imgT in enumerate(x_train_flatten):

        # Compute L1 (Manhattan) distance
        difference = np.abs(img - imgT)
        score = np.sum(difference)

        # Store (distance, label) pairs
        predictions.append((score, y_train[idx][0]))

    # Sort by distance (ascending order)
    predictions = sorted(predictions, key=lambda x: x[0])

    # Get k nearest neighbors
    top_k_predictions = predictions[0:k]

    # Extract labels from top k neighbors
    predLabels = [label for _, label in top_k_predictions]

    # Majority vote: find the most common label
    predictedLabel = max(set(predLabels), key=predLabels.count)

    return predictedLabel


def main():
    """
    Main function to evaluate Nearest Neighbor and K-Nearest Neighbors classifiers
    on the CIFAR-10 dataset.
    """

    # ========== Load Dataset ==========

    print("=" * 60)
    print("LOADING CIFAR-10 DATASET")
    print("=" * 60 + "\n")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Display dataset dimensions
    print(f"Training images:   {x_train.shape}")
    print(f"Training labels:   {y_train.shape}")
    print(f"Test images:       {x_test.shape}")
    print(f"Test labels:       {y_test.shape}")

    # ========== Explore Dataset ==========

    print("\n" + "=" * 60)
    print("FIRST 10 TEST IMAGES - LABELS")
    print("=" * 60 + "\n")

    for i in range(10):
        label_index = y_test[i][0]
        print(f"Image {i}: Label {label_index} ({DICT_CLASSES[label_index]})")

    # ========== Preprocessing ==========

    print("\n" + "=" * 60)
    print("PREPROCESSING: FLATTENING IMAGES")
    print("=" * 60 + "\n")

    # Flatten images from 32×32×3 to 3072-dimensional vectors
    x_train_flatten = np.float64(
        x_train.reshape(x_train.shape[0], 32 * 32 * 3))
    x_test_flatten = np.float64(x_test.reshape(x_test.shape[0], 32 * 32 * 3))

    print(f"Flattened training images: {x_train_flatten.shape}")
    print(f"Flattened test images:     {x_test_flatten.shape}")

    # ========== Nearest Neighbor Classification ==========

    print("\n" + "=" * 60)
    print("NEAREST NEIGHBOR CLASSIFICATION")
    print("=" * 60 + "\n")

    numberOfCorrectPredictedImages = 0
    num_images_to_test = 200

    for idx, img in enumerate(x_test_flatten[0:num_images_to_test]):

        print(f"Classifying test image {idx}/{num_images_to_test}...")

        # Predict using Nearest Neighbor
        predictedLabel = predictLabelNN(x_train_flatten, y_train, img)

        # Compare with ground truth
        ground_truth_label = y_test[idx][0]

        if predictedLabel == ground_truth_label:
            numberOfCorrectPredictedImages += 1

        # Display result
        match_status = "✓" if predictedLabel == ground_truth_label else "✗"
        print(
            f"  {match_status} Predicted: {predictedLabel} ({DICT_CLASSES[predictedLabel]}) | "
            f"True: {ground_truth_label} ({DICT_CLASSES[ground_truth_label]})"
        )

    # ========== Results ==========

    accuracy = 100 * numberOfCorrectPredictedImages / num_images_to_test

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"Correct predictions: {numberOfCorrectPredictedImages}/{num_images_to_test}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)

    return


if __name__ == '__main__':
    main()
