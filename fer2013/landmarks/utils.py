#!/usr/bin/env oython

import dlib
import numpy as np
from os import getcwd
from os import path as op


def load_images(num_ims, fname):
    pix = []
    with open(fname, 'r') as f:
        for _ in range(num_ims):
            line = f.readline()
            pix.append(np.array(map(int, line.split(','))).astype(np.uint8))
    return np.array(pix)

def reformat(data):
    x = data[:,1:]
    c = data[:,0]
    k = max(c) + 1
    n = len(x)
    y = np.zeros((n, k))
    y[np.arange(n), c] = 1
    return x, y, c

def get_landmarks(X, detect, predict, dims=(48,48)):
    n = len(X)
    bad = 0
    idx = 0
    Xnew = np.zeros((n, 2*68))
    for pix in X:
        img = pix.reshape(dims)
        dets = detect(img, 1)
        if len(dets) <= 0:
            bad += 1
            continue
        rect = dets[0]
        shape = predict(img, rect)
        for i, p in enumerate(shape.parts()):
            Xnew[idx,2*i] = float((p.x - rect.left()))/ (rect.right() - rect.left())
            Xnew[idx,2*i+1] = float((p.y - rect.top()))/ (rect.bottom() - rect.top())
        idx += 1
    print "bad %i" % bad
    return Xnew[:idx,:]


if __name__=="__main__":
    FER = op.dirname(getcwd())
    DLIB = op.join(op.dirname(FER), "dlib")
    trained_pred = op.join(DLIB, "python_examples/shape_predictor_68_face_landmarks.dat")

#    traindat = np.loadtxt(op.join(FER, "train.csv"), dtype=np.uint8, delimiter=',')
#    data = np.loadtxt(op.join(FER, "test1.csv"), dtype=np.uint8, delimiter=',')
    data = load_images(100, op.join(FER, "test1.csv"))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(trained_pred)

    X, Y, C = reformat(data)
    Xnew = get_landmarks(X[:100,:], detector, predictor)
    print Xnew
