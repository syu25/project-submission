import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn


def loadData(filename = 'creditcard.csv'):
    data = pd.read_csv(filename)            #load dataset file as pandRRRas dataframe
    return data                             #returns pandas dataframe

def getTargetValues(arr):                       #splits the dataset into input and output
                                                #arguements: array holding dataset
    dataList = np.ndarray.tolist(arr.T)         #convert transposed array to list (in order to use pop())
    targetValues = np.asarray(dataList.pop(len(dataList) - 1))  #remove target values from dataset
    #returns: array of input, array of target values
    return np.asarray(dataList, dtype=np.float64).T, np.asarray(targetValues, dtype=np.float64)

def splitData(arr):                         #splits data to training, validation and testing sets
                                            #arguements: array holding dataset
    dataList = np.ndarray.tolist(arr)       #convert array to list
    eightyPercent = int((len(dataList))*0.8)    #80% of data to be in training
    tenPercent = int((len(dataList))*0.1)       #10% to in validation
    training = []                               #remaining 10% in dataList will be used for testing
    validation = []

    for i in range(eightyPercent + 1):          #randomly select 80% of instances
        randomLine = random.randint(0, len(dataList) - 1)
        training.append(dataList.pop(randomLine))

    for i in range(tenPercent + 1):             #randomly select 10% of instances
        randomLine = random.randint(0, len(dataList) - 1)
        validation.append(dataList.pop(randomLine))

    #returns training, validation and testing arrays
    return np.asarray(training), np.asarray(validation), np.asarray(dataList)

def run_LR():

    data = pd.DataFrame.to_numpy(loadData())                    #convert dataset to from pandas dataframe to a numpy array
    data = np.delete(data, 0, 1)
    training, validation, testing = splitData(data)                         #split up  the data to 3 (80% for training, 10% for validation, 10% for testing

    trainingX, trainingY = getTargetValues(training)                            #split training set to input and output
    trainingX = preprocessing.scale(X = trainingX, axis = 0, with_mean = True, with_std = True)                                        #normalize training set (function normalize() returns normalized and transposed array)

    validationX , validationY = getTargetValues(validation)
    validationX = preprocessing.scale(X = validationX, axis = 0, with_mean = True, with_std = True)

    learningRate = 0.045
    batchSize = 512
    epochCount = 75
    inputDimension = trainingX.shape[1]

    model = model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim = inputDimension, activation='sigmoid', activity_regularizer = tf.keras.regularizers.l2(0.01)))
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = learningRate), loss = "binary_crossentropy", metrics = ['accuracy'])


    fraudCount = 0
    for i in range(len(trainingY)):
        if trainingY[i] == 1:
            fraudCount = fraudCount + 1

    classWeights = {0: len(trainingY) / (2 * (len(trainingX) - fraudCount)),
                    1: len(trainingY) / (2 * fraudCount)}

    hist = model.fit(trainingX, trainingY, batch_size = batchSize, epochs = epochCount, validation_data = (validationX, validationY), shuffle = True, class_weight = classWeights)

    testingX, testingY = getTargetValues(testing)              #repeat process for testing data
    testingX = preprocessing.scale(X = testingX, axis = 0, with_mean = True, with_std = True)

    predictions = model.predict(testingX, verbose = 0)

    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            predictions[i] = 1

        else:
            predictions[i] = 0

    m = tf.keras.metrics.Accuracy()
    _ = m.update_state(testingY, predictions)
    accuracy = m.result().numpy()

    precision, recall, fBetaScore, support = metrics.precision_recall_fscore_support(testingY, predictions)

    line1 = ("{}{}".format("n = ", len(testingY)))
    line2 = ("{}{}".format("Accuracy = ", accuracy))
    line3 = ("{}{}".format("Precision = ", precision))
    line4 = ("{}{}".format("Recall = ", recall))
    line5 = ("{}{}".format("F1 Score = ", fBetaScore))

    retlist = [line1, line2, line3, line4, line5]

    epochs = list(range(epochCount))
    plt.figure(figsize=(5, 4), dpi=70)
    plt.scatter(epochs, hist.history['val_loss'], s = 10, marker = "s", label = 'Validation data', c = 'b')        #plot validation loss versus epochs
    plt.scatter(epochs, hist.history['loss'], s = 10, marker = "o", label = 'Training data', c = 'r')            #plot training loss versus epchs
    plt.legend(loc='upper right');
    plt.xlabel("# of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss versus Epochs")
    plt.savefig("loss_vs_epoch")

    confusionMatrix = metrics.confusion_matrix(testingY, predictions)
    plt.figure(figsize=(5, 4), dpi=70)
    heatMatrix = sn.heatmap(confusionMatrix, annot = True)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig("actual_vs_predicted")

    plt.figure(figsize=(5, 4), dpi=70)
    fpr, tpr, _ = metrics.roc_curve(testingY, predictions)
    auc_keras = metrics.auc(fpr, tpr)


    plt.plot([0, 1], [0, 1], 'k--', c = 'k')
    plt.plot(fpr, tpr, marker='.', c = 'r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("false_true_rates")
    #plt.show()

    return retlist
