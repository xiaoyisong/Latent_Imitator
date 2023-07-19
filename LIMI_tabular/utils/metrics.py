import os

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score
from scipy.ndimage import gaussian_filter


def p4(targets, pred):
    TP = (pred[pred == targets] == 1).sum()  # TP
    FN = (pred[pred != targets] == 0).sum()  # FN
    FP = (pred[pred != targets] == 1).sum()  # FP
    TN = (pred[pred == targets] == 0).sum()  # TN
    return TP, FN, FP, TN


### the metric of aif360
def SPD_DI(domain, targets, pred):
    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)

    predict_y1_g0 = (pred[g0] == 1).sum()
    predict_y1_g1 = (pred[g1] == 1).sum()
    print('predict_y1_g0', predict_y1_g0)
    spd = predict_y1_g0 / len(g0) - predict_y1_g1 / len(g1)
    di = predict_y1_g0 / len(g0) / predict_y1_g1 / len(g1)
    return spd, di, abs(spd), abs(di)


def EOD(domain, targets, pred):
    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)

    eod = np.abs(
        recall_score(targets[g0], pred[g0]) - recall_score(targets[g1], pred[g1])
    )
    TP0, FN0, FP0, TN0 = p4(targets[g0], pred[g0])
    TP1, FN1, FP1, TN1 = p4(targets[g1], pred[g1])
    eod = TP0 / (TP0 + FN0) - TP1 / (TP1 + FN1)
    return eod, abs(eod)


def AOD(domain, targets, pred):
    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)
    TP0, FN0, FP0, TN0 = p4(targets[g0], pred[g0])
    TP1, FN1, FP1, TN1 = p4(targets[g1], pred[g1])
    aod = 0.5 * (
        (TP0 / (TP0 + FN0) - TP1 / (TP1 + FN1))
        + (FP0 / (FP0 + TN0) - FP1 / (FP1 + TN1))
    )
    aod_abs = 0.5 * (
        abs(TP0 / (TP0 + FN0) - TP1 / (TP1 + FN1))
        + abs(FP0 / (FP0 + TN0) - FP1 / (FP1 + TN1))
    )
    return aod, aod_abs


def ERD(domain, targets, pred):
    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)
    TP0, FN0, FP0, TN0 = p4(targets[g0], pred[g0])
    TP1, FN1, FP1, TN1 = p4(targets[g1], pred[g1])
    erd = (FP0 / (FP0 + TN0) - FP1 / (FP1 + TN1)) + (
        FN0 / (TP0 + FN0) - FN1 / (TP1 + FN1)
    )
    erd_abs = abs(FP0 / (FP0 + TN0) - FP1 / (FP1 + TN1)) + abs(
        FN0 / (TP0 + FN0) - FN1 / (TP1 + FN1)
    )
    return erd, erd_abs
