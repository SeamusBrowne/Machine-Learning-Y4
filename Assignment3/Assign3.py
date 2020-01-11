import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
from collections import defaultdict


def num_coefficients_2(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t += 1
    return t


def calculate_model_function(deg, shape, p):
    result = 0
    k = 0

    for n in range(deg + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for l in range(n + 1):
                    if i + j + l == n:
                        result += p[k] * (shape[0] ** i) * (shape[1] ** j) * (shape[2] ** l)
                        k += 1
    return result


def linearization(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    J = np.zeros((1, len(p0)))
    epsilon = 1e-6

    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di

    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


def read_file(file):
    data = pd.read_csv(file)
    attributes = data[["cut", "color", "clarity"]]
    shape = data[["depth", "table", "carat"]]
    predict = data[["price"]]

    dictionary = defaultdict(lambda: (list(), list()))

    for attribute, shape, predict in zip(attributes.values, shape.values, predict.values):
        dictionary[tuple(attribute)][0].append(shape)
        dictionary[tuple(attribute)][1].append(predict)

    reduced_data = dict()
    data_length = 800

    for k, v in dictionary.items():
        if len(v[0]) > data_length:
            reduced_data[k] = v

    return reduced_data


def k_fold(shape, predict):
    best_overall = ()
    split = 6
    best_overall_mean = -1

    k_fold_model = model_selection.KFold(n_splits=split)

    for i, (train_i, index) in enumerate(k_fold_model.split(shape, predict)):
        optimal_degree = 0
        optimal_mean = -1

        for degree in range(1, 4):
            train_shape = shape.iloc[train_i].tolist()
            train_predict = predict.iloc[train_i].tolist()
            test_shape = shape.iloc[index].tolist()
            test_predict = predict.iloc[index].tolist()

            regression = regression_model(degree, train_shape, train_predict)
            data_changes = list()

            for feature, target in zip(test_shape, test_predict):
                predicted = calculate_model_function(degree, feature, regression)
                data_changes.append(abs(predicted - target))

            mean_value = np.mean(data_changes)

            if optimal_mean == -1 or mean_value < optimal_mean:
                optimal_degree = degree
                optimal_mean = mean_value

                if best_overall_mean == -1 or mean_value < best_overall_mean:
                    best_overall = (optimal_degree, regression)
                    best_overall_mean = optimal_mean

        optimal_mean = str(round(optimal_mean, 3))
        print("Split " + str(i + 1) + "/" + str(split) +
              "\n\t Degree with best polynominal: " + str(optimal_degree) +
              "\n\t Mean price difference : " + str(optimal_mean))

    return best_overall


def regression_model(degree, shape, predict):
    max_iter = 10
    p0 = np.zeros(num_coefficients_2(degree))
    f0 = np.zeros(len(shape))
    J = np.zeros((len(f0), len(p0)))

    for i in range(max_iter):
        i = 0
        prediction = np.zeros(len(f0))

        for feature, target in zip(shape, predict):
            f0[i], J[i] = linearization(degree, feature, p0)
            prediction[i] = target[0]
            i += 1

        dp = calculate_update(prediction, f0, J)
        p0 += dp

    return p0


def display_data(shapes, predict, optimal):
    print("Displaying Graph with the acquired results: ")
    prediction = list()

    for shape in shapes:
        prediction.append(calculate_model_function(optimal[0], shape, optimal[1]))

    plt.close()
    plt.ylabel("Correct Diamond Price")
    plt.xlabel("Predicted Diamond Price")
    plt.plot(prediction, predict, 'r,')
    plt.show()


def main():
    data = read_file("diamonds.csv")
    print("\n Regression performance:")

    for i, splits in enumerate(data.values()):
        print("Split " + str(i + 1) + "/" + str(len(data.values())) + ":")
        shape = np.array(splits[0])
        predict = np.array(splits[1])
        regression = regression_model(2, shape, predict)
        print(regression)

    print("\nK-Fold Performance and Graphical Representation: ")

    for i, splits in enumerate(data.values()):
        print("Subset " + str(i + 1) + "/" + str(len(data.values())) + ":")
        shape = pd.Series(splits[0])
        predict = pd.Series(splits[1])
        optimal = k_fold(shape, predict)

        shape = np.array(splits[0])
        predict = np.array(splits[1])
        display_data(shape, predict, optimal)


main()
