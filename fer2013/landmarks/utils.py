#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def make_heatmap(data, rlabels, clabels, rname, cname): # data should already be in matrix form
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(clabels, minor=False)
    ax.set_yticklabels(rlabels, minor=False)
    ax.set_xlabel(cname)
    ax.set_ylabel(rname)
    plt.show()


if __name__=="__main__":
    data = np.random.rand(10, 10)
    rlabels = np.arange(0.1, 1.1, 0.1)
    clabels = range(10)
    make_heatmap(data, rlabels, clabels, "Rows", "Columns")
