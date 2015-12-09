#!/usr/bin/env python

from utils import *

import os
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


def load_features(fname):
    data = np.loadtxt(fname, delimiter=",")
    x = data[:,1:]
    c = data[:,0].astype(np.uint8)
    k = max(c) + 1
    n = x.shape[0]
    y = np.zeros((n, k))
    y[np.arange(n), c] = 1
    return x, y, c

def make_grid(grid_scores, numc, numgam):
    idx = 0
    data = np.zeros((numc, numgam))
    for i in range(numc):
        for j in range(numgam):
            data[i,j] = grid_scores[idx][1]
            idx += 1
    return data


if __name__=="__main__":
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    x, y, c = load_features(os.path.join(DATA_DIR, "test1new.csv"))
    xt, yt, ct = load_features(os.path.join(DATA_DIR, "test2new.csv"))
    print x.shape, y.shape, c.shape

    cvals = [1, 5, 1e1, 5e1, 1e2, 5e2]
    gamvals = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
    param_grid = {'C': cvals,'gamma': gamvals, }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf = clf.fit(x, c)
    print "score on training", clf.score(x, c)
    print "score on test", clf.score(xt, ct)

    grid = make_grid(clf.grid_scores_, len(cvals), len(gamvals))
    print grid
    make_heatmap(grid, cvals, gamvals, "C", "gamma")
