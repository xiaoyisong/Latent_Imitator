import numpy as np
import sys

sys.path.append("../")


def credit_data(
    path='../datasets/credit_sample',
):
    """
    Prepare the data of dataset German Credit
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

    input_shape = (None, 20)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def credit_predict_data(
    paths=["../datasets/credit_sample"],
):
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
                X.append(L[:20])

    X = np.array(X, dtype=float)

    input_shape = (None, 20)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def credit_eval_data(
    path="../datasets/credit_sample",
    protected_index=8,  # gender
):
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

    input_shape = (None, 20)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


X, _Y, input_shape, nb_classes = credit_data()
Y = []
for ind in range(0, len(_Y)):
    if _Y[ind][0] == 1:
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y, dtype=float)
# print(Y)
# for german credit data, gender(8) and age(12) are protected attributes in 24 features
protected_attribs = [8, 12]
# for test it need add 1
