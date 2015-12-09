#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    # Helper function to plot a gallery of portraits
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

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
    fig.show()


if __name__=="__main__":
    data = np.random.rand(10, 10)
    rlabels = np.arange(0.1, 1.1, 0.1)
    clabels = range(10)
    make_heatmap(data, rlabels, clabels, 'rows', 'columns')
