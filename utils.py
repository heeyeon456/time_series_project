import numpy as np
from scipy.stats import pearsonr


def calcRMSE(true, pred):
    return np.sqrt(np.square((true-pred)).mean())


def calcMAE(true, pred):
    return np.abs((true-pred)).mean()


def calcMAPE(true, pred, epsilon=0.1):
    true += epsilon
    return np.nanmean(np.abs((true - pred) / true)) * 100


def calcPRMSE(true, pred):
    #return calcRMSE(true, pred) * 100 / np.nanmean(true[true!=0])
    return calcRMSE(true, pred) * 100 / np.percentile(true, 99.5, interpolation='nearest')


def calcPRMSE_mean(true, pred):
    return calcRMSE(true, pred) * 100 / np.nanmean(true[true!=0])


def calcSMAPE(true, pred):
    delim = (np.abs(true) + np.abs(pred)) / 2.0
    return np.nanmean(np.abs((true - pred) / delim)) * 100


def calcCorr(true, pred):
    r, p = pearsonr(np.squeeze(true).flatten(), np.squeeze(pred).flatten())
    return r*100, p
