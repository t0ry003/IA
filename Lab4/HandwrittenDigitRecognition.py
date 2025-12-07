import tensorflow as tf
from tensorflow.keras import layers, models


def baseline_model(num_pixels, num_classes):
    """
    Creates a simple Multi-Layer Perceptron (MLP) model.

    Args:
        num_pixels (int): Number of input pixels (features).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: The compiled MLP model.
    """

    # TODO - Application 1 - Step 6a - Initialize the sequential model
    model = models.Sequential()

    # TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons
    model.add(layers.Dense(8, input_dim=num_pixels,
                           kernel_initializer='normal', activation='relu'))
    # TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal',
                           activation='softmax'))

    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def trainAndPredictMLP(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates an MLP model on the given dataset.

    Args:
        x_train (numpy.ndarray): Training data images.
        y_train (numpy.ndarray): Training data labels.
        x_test (numpy.ndarray): Test data images.
        y_test (numpy.ndarray): Test data labels.
    """

    # TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')

    # TODO - Application 1 - Step 4 - Normalize the input values
    x_train = x_train / 255
    x_test = x_test / 255

    # TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
    model = baseline_model(num_pixels, 10)

    # TODO - Application 1 - Step 7 - Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,
              batch_size=200, verbose=2)
    # TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: {:.2f}".format(100-scores[1]*100))

    return


def CNN_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: The compiled CNN model.
    """

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = models.Sequential()

    # TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))

    # TODO - Application 2 - Step 5c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # TODO - Application 2 - Step 5d - Define the Dropout layer
    model.add(layers.Dropout(0.2))

    # TODO - Application 2 - Step 5e - Define the flatten layer
    model.add(layers.Flatten())

    # TODO - Application 2 - Step 5f - Define a dense layer of size 128
    model.add(layers.Dense(128, activation='relu'))

    # TODO - Application 2 - Step 5g - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # TODO - Application 2 - Step 5h - Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def trainAndPredictCNN(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates a CNN model on the given dataset.

    Args:
        x_train (numpy.ndarray): Training data images.
        y_train (numpy.ndarray): Training data labels.
        x_test (numpy.ndarray): Test data images.
        y_test (numpy.ndarray): Test data labels.
    """

    # TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]
    x_train = x_train.reshape(
        (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype('float32')
    x_test = x_test.reshape(
        (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype('float32')

    # TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # TODO - Application 2 - Step 5 - Call the CNN_model function
    model = CNN_model((x_train.shape[1], x_train.shape[2], 1), num_classes)

    # TODO - Application 2 - Step 6 - Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,
              batch_size=200, verbose=2)

    # TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Error: {:.2f}".format(100-scores[1]*100))

    return


def main():
    """
    Main function to load data and run the training/prediction processes.
    """

    # TODO - Application 1 - Step 1 - Load the MNIST dataset in Tensorflow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    trainAndPredictMLP(x_train, y_train, x_test, y_test)
    # TODO - Application 2 - Step 1 - Train and predict on a CNN - Call the trainAndPredictCNN function
    trainAndPredictCNN(x_train, y_train, x_test, y_test)
    return


if __name__ == '__main__':
    main()
