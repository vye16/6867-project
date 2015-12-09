#!/usr/bin/env python

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


def load_features(fname):
    data = np.loadtxt(fname, delimiter=",")
    x = data[:,1:]
    c = data[:,0].astype(np.uint8)
    k = max(c) + 1
    n = x.shape[0]
    y = np.zeros((n, k))
    y[np.arange(n), c] = 1
    return x, y, c


if __name__=="__main__":
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    x, y, c = load_features(os.path.join(DATA_DIR, "test1new.csv"))
    xt, yt, ct = load_features(os.path.join(DATA_DIR, "test2new.csv"))
    print x.shape, y.shape, c.shape

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf = clf.fit(x, c)
    print "score on training", clf.score(x, c)
    print "score on test", clf.score(xt, ct)
