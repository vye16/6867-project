#!/usr/bin/env oython

import dlib
import numpy as np
from os import getcwd
from os import path as op
import matplotlib.pyplot as plt

# DIRNAMES
FER = op.dirname(getcwd())
DLIB = op.join(op.dirname(FER), "dlib")

def load_images(num_ims, fname):
    pix = []
    with open(fname, 'r') as f:
        for _ in range(num_ims):
            line = f.readline()
            pix.append(np.array(map(int, line.split(','))).astype(np.uint8))
    return np.array(pix)

def reformat(data):
    x = data[:,1:]
    c = data[:,0:1]
    k = max(c) + 1
    n = len(x)
    y = np.zeros((n, k))
    y[np.arange(n), c] = 1
    return x, y, c

def get_landmarks(X, detect, predict, dims=(48,48)):
    n = len(X)
    bad = 0
    idx = 0
    good = []
    Xnew = np.zeros((n, 2*68))
    imovers = []
    for i, pix in enumerate(X):
        img = pix.reshape(dims)
        dets = detect(img, 1)
        if len(dets) <= 0:
            bad += 1
            continue
        rect = dets[0]
        shape = predict(img, rect)
        imovers.append((img, rect, shape))
        for j, p in enumerate(shape.parts()):
#            Xnew[idx,2*i] = float((p.x - rect.left()))/ (rect.right() - rect.left())
#            Xnew[idx,2*i+1] = float((p.y - rect.top()))/ (rect.bottom() - rect.top())
            Xnew[idx,2*j] = p.x
            Xnew[idx,2*j+1] = p.y
        good.append(i)
        idx += 1
    print "bad %i" % bad
    return Xnew[:idx,:], np.array(good), imovers

def write_to_file(trans, fname):
    with open(fname, "w") as f:
        for row in trans:
            f.write(','.join(map(str, row)) + '\n')

if __name__=="__main__":
    trained_pred = op.join(DLIB, "python_examples/shape_predictor_68_face_landmarks.dat")

    traindat = np.loadtxt(op.join(FER, "train.csv"), dtype=np.uint8, delimiter=',')
    test1dat = np.loadtxt(op.join(FER, "test1.csv"), dtype=np.uint8, delimiter=',')
    test2dat = np.loadtxt(op.join(FER, "test2.csv"), dtype=np.uint8, delimiter=',')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(trained_pred)

    dats = [traindat, test1dat, test2dat]
#    dats = [load_images(20, op.join(FER, "test1.csv"))]
    trans = []
    for data in dats:
        X, _, C = reformat(data)
        Xnew, good, imovers = get_landmarks(X, detector, predictor)
        C = C[good]
        print "Xnew shape", Xnew.shape 
        trans.append(np.hstack((C, 10 * Xnew)))
    
    write_to_file(trans[0], "trainnew.csv")
    write_to_file(trans[1], "test1new.csv")
    write_to_file(trans[2], "test2new.csv")
