import numpy as np
import sys

sys.path.append("../")


"""
 feature_names = ['REGION' 'AGE' 'SEX' 'RACE' 'MARRY' 'FTSTU' 'ACTDTY' 'HONRDC' 'RTHLTH'
 'MNHLTH' 'CHDDX' 'ANGIDX' 'MIDX' 'OHRTDX' 'STRKDX' 'EMPHDX' 'CHBRON'
 'CHOLDX' 'CANCERDX' 'DIABDX' 'JTPAIN' 'ARTHDX' 'ARTHTYPE' 'ASTHDX'
 'ADHDADDX' 'PREGNT' 'WLKLIM' 'ACTLIM' 'SOCLIM' 'COGLIM' 'DFHEAR42'
 'DFSEE42' 'ADSMOK42' 'PCS42' 'MCS42' 'K6SUM42' 'PHQ242' 'EMPST' 'POVCAT'
 'INSCOV']
"""
### REGION,AGE,SEX,RACE,MARRY,FTSTU,ACTDTY,HONRDC,RTHLTH,MNHLTH,CHDDX,ANGIDX,MIDX,OHRTDX,STRKDX,EMPHDX,CHBRON,CHOLDX,CANCERDX,DIABDX,JTPAIN,ARTHDX,ARTHTYPE,ASTHDX,ADHDADDX,PREGNT,WLKLIM,ACTLIM,SOCLIM,COGLIM,DFHEAR42,DFSEE42,ADSMOK42,PCS42,MCS42,K6SUM42,PHQ242,EMPST,POVCAT,INSCOV


def meps_data(path="../datasets/meps"):
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

    input_shape = (None, 40)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def meps_predict_data(
    paths=["../datasets/meps"],
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
                X.append(L[:40])

    X = np.array(X, dtype=float)

    input_shape = (None, 40)
    nb_classes = 2

    return X, None, input_shape, nb_classes


def meps_eval_data(
    path="../datasets/meps",
    protected_index=2,  # sex
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

    input_shape = (None, 40)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


X, _Y, input_shape, nb_classes = meps_data()
# print(X.shape)
Y = []
for ind in range(0, len(_Y)):
    if _Y[ind][0] == 1:
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y, dtype=float)

# for meps data, sex(2) is protected attributes in 40 features
# (15675, 40)
protected_attribs = [2]
