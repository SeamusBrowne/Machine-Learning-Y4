import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, linear_model, svm


# Data is inputed from the file and the data is extracted.
def read_file(file, length):
    data = pd.read_csv(file)
    label, image = data["label"].iloc[:length], data.drop(columns="label").iloc[:length]

    print("Number of Sneakers: " + str(len(label[label == 0])))
    print("Number of Ankle Boots: " + str(len(label[label == 1])))

    # Displays one of each of images in sneaker and ankle boot labels
    # Plots the numeric data in a 28 x 28 grid to display images and converts to grayscale
    for shoe in range(0, 2):
        i = next(x for x in range(len(label)) if label.iloc[x] == shoe)
        display_image = image.iloc[i].to_numpy().reshape(28, 28)
        plt.imshow(display_image, cmap="gray")
        plt.show()

    return label, image


# Creates a k-fold that splits the data into smaller sections
def k_fold(label, data, split, classification):
    train_runtimes = []
    predict_runtimes = []
    model_accuracy = []

    k_fold_model = model_selection.KFold(n_splits=split)
    for i, (train_i, test_i) in enumerate(k_fold_model.split(data)):
        print("\tK-fold " + str(i + 1) + " / " + str(split))
        train, training_label = data.iloc[train_i], label.iloc[train_i]
        test, testing_label = data.iloc[test_i], label.iloc[test_i]

        # All timers for the duration a process takes to complete ie. time
        # taken to train data, time for prediction
        train_timer = time()
        classification.fit(train, training_label)
        train_runtimes.append(time() - train_timer)
        prediction_timer = time()
        prediction = classification.predict(test)
        predict_runtimes.append(time() - prediction_timer)

        accuracy = round(metrics.accuracy_score(testing_label, prediction) * 100, 2)
        model_accuracy.append(accuracy)
        print("\tAccuracy: " + str(accuracy) + "%")
        print("\tConfusion Matrix " + str(i + 1) + " Results:")
        confusion_matrix = metrics.confusion_matrix(testing_label, prediction)
        for matrix in confusion_matrix:
            print("\t", matrix)

    #Displays runtimes of each processes length.
    print("\tAverage training runtime: " + str(np.round(np.average(train_runtimes), 2)) + " seconds")
    print("\tMinimum training runtime: " + str(np.round(np.min(train_runtimes), 2)) + " seconds")
    print("\tMaximum training runtime: " + str(np.round(np.max(train_runtimes), 2)) + " seconds")

    print("\tAverage prediction runtime: " + str(np.round(np.average(predict_runtimes), 3)) + " seconds")
    print("\tMinimum prediction runtime: " + str(np.round(np.min(predict_runtimes), 3)) + " seconds")
    print("\tMaximum prediction runtime: " + str(np.round(np.max(predict_runtimes), 3)) + " seconds")

    return np.average(model_accuracy)


def main(length):
    label, data = read_file("product_images.csv", length)

    #Perceptron and Linear model initilisation. K-fold split the data into 12 splits for both models.
    #Finds the average mean accuracy of the models.
    print("Perceptron Model:")
    clf = linear_model.Perceptron()
    accuracy_perceptron = k_fold(label, data, 12, clf)
    print("Linear Model:")
    clf1 = svm.SVC(kernel="linear")
    accuracy_linear = k_fold(label, data, 12, clf1)
    print("Final percentage results of Models: ")
    print("Average Perceptron accuracy: " + str(accuracy_perceptron))
    print("Average Linear accuracy: " + str(accuracy_linear))


main(2000)
