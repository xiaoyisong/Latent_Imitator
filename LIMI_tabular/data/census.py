import numpy as np
import sys

sys.path.append("../")


def census_data(
    path="../datasets/census",
):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(path, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def census_predict_data(
    paths=["../datasets/census"],
):
    """
    Prepare the data of dataset Census Income
    there is no label of the dataset
    :return: X, Y, input shape and number of classes
    """
    X = []

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        with open(path, "r") as ins:
            i = 0
            for line in ins:
                line = line.strip()
                line1 = line.split(",")
                if i == 0:
                    i += 1
                    continue
                # L = map(int, line1[:-1])
                L = [int(i) for i in line1]
                X.append(L[:13])

    X = np.array(X, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def census_eval_data(
    path="../datasets/census",
    protected_index=8,  # gender
):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(path, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            if i == 0:
                i += 1
                continue
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            Y.append([int(line1[-1]), int(L[protected_index])])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


X, _Y, input_shape, nb_classes = census_data()
# print(X.shape)
Y = []
for ind in range(0, len(_Y)):
    if _Y[ind][0] == 1:
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y, dtype=float)
# print(Y)
# for census income data, age(0), race(7) and gender(8) are protected attributes in 12 features
protected_attribs = [0, 7, 8]
