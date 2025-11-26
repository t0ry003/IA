# Application 1 - Step 1 - Import the dependencies
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import fashion_mnist
import cv2
#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################


def summarizeLearningCurvesPerformances(histories, accuracyScores):

    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='green', label='train')
        pyplot.plot(histories[i].history['val_loss'],
                    color='red', label='test')

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'],
                    color='green', label='train')
        pyplot.plot(histories[i].history['val_accuracy'],
                    color='red', label='test')

        # print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(
        accuracyScores) * 100, np.std(accuracyScores) * 100, len(accuracyScores)))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def prepareData(trainX, trainY, testX, testY):

    # TODO - Application 1 - Step 4a - reshape the data to be of size [samples][width][height][channels]
    trainX = trainX.reshape(
        (trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)).astype('float32')
    testX = testX.reshape(
        (testX.shape[0], testX.shape[1], testX.shape[2], 1)).astype('float32')

    # TODO - Application 1 - Step 4b - normalize the input values
    trainX = trainX / 255.0
    testX = testX / 255.0

    # TODO - Application 1 - Step 4c - Transform the class labels (a vector of integers) into a binary matrix
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    return trainX, trainY, testX, testY
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):

    # TODO - Application 1 - Step 6a - Initialize the sequential model
    model = models.Sequential()

    # TODO - Application 1 - Step 6b - Create the first hidden layer as a convolutional layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))

    # TODO - Application 1 - Step 6c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # TODO - Application 1 - Step 6d - Define the flatten layer
    model.add(layers.Flatten())

    # TODO - Application 1 - Step 6e - Define a dense layer of size 16
    model.add(layers.Dense(16, activation='relu'))

    # TODO - Application 1 - Step 6f - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # TODO - Application 1 - Step 6g - Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):

    # TODO - Application 1 - Step 6 - Call the defineModel function
    # Step  6:  Build  the  model  –  We  will  construct  our  model  in  a  function  denoted  def defineModel(input_shape, num_classes)
    model = defineModel((trainX.shape[1], trainX.shape[2], 1), trainY.shape[1])

    # TODO - Application 1 - Step 7 - Train the model
    model.fit(trainX, trainY, epochs=10, batch_size=32)

    # TODO - Application 1 - Step 8 - Evaluate the model
    loss, accuracy = model.evaluate(testX, testY)
    print(f"Test accuracy: {accuracy:.4f}")

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5

    accuracyScores = []
    histories = []

    # Application 2 - Step 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(trainX):

        # TODO - Application 2 - Step 3 - Select data for train and validation
        trainX_i = trainX[train_idx]
        trainY_i = trainY[train_idx]
        valX_i = trainX[val_idx]
        valY_i = trainY[val_idx]

        # TODO - Application 2 - Step 4 - Build the model - Call the defineModel function
        model = defineModel(
            (trainX.shape[1], trainX.shape[2], 1), trainY.shape[1])

        # TODO - Application 2 - Step 5 - Fit the model
        history = model.fit(trainX_i, trainY_i, validation_data=(
            valX_i, valY_i), epochs=10, batch_size=32, verbose=0)
        # TODO - Application 2 - Step 6 - Save the training related information in the histories list
        histories.append(history)

        # TODO - Application 2 - Step 7 - Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(testX, testY, verbose=0)
        print(f"Test accuracy: {accuracy:.4f}")

        # TODO - Application 2 - Step 8 - Save the accuracy in the accuracyScores list
        accuracyScores.append(accuracy)

    return histories, accuracyScores
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    # TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    # TODO - Application 1 - Step 3 - Print the size of the train/test dataset
    print("Train data shape: ", trainX.shape, trainY.shape)
    print("Test data shape: ", testX.shape, testY.shape)

    # TODO - Application 1 - Step 4 - Call the prepareData method
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)

    # TODO - Application 1 - Step 5 - Define, train and evaluate the model in the classical way
    defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)

    # TODO - Application 2 - Step 1 - Define, train and evaluate the model using K-Folds strategy
    histories, accuracyScores = defineTrainAndEvaluateKFolds(
        trainX, trainY, testX, testY)
    summarizeLearningCurvesPerformances(histories, accuracyScores)

    # TODO - Application 2 - Step9 - System performance presentation
    print(f"Average accuracy: {np.mean(accuracyScores):.4f}")

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
