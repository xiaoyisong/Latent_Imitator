import numpy as np
import sys

sys.path.append("../")


def bank_data(path="../datasets/bank"):
    """
    Prepare the data of dataset Bank Marketing
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0
    with open(path, "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
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

    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def bank_predict_data(
    paths=["../datasets/bank"],
):
    """
    Prepare the data of dataset Bank Marketing
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
                L = [int(i) for i in line1]
                X.append(L[:16])

    X = np.array(X, dtype=float)

    input_shape = (None, 16)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def bank_eval_data(
    path="../datasets/bank",
    protected_index=0,  # age
):
    """
    Prepare the data of dataset Bank Marketing
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

    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


X, _Y, input_shape, nb_classes = bank_data()
Y = []
for ind in range(0, len(_Y)):
    if _Y[ind][0] == 1:
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y, dtype=float)
# print(Y)
# for bank marketing data, age(0) is the protected attribute in 16 features
protected_attribs = [0]
