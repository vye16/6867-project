#!/usr/bin/env oython

from utils import *
from sklearn import cross_validation as crossval
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


if __name__=="__main__":

    DATA_DIR = os.path.dirname(os.getcwd())
    data = np.loadtxt(os.path.join(DATA_DIR, "train.csv"), dtype=np.int, delimiter=",")
    datat = np.loadtxt(os.path.join(DATA_DIR, "test2.csv"), dtype=np.int, delimiter=",")
    N = len(data)
    print N
    data1 = data[8*N/10:9*N/10,:]
    data2 = data[9*N/10:,:]

    # X = pca extraction (includes training but not validation)
    # Xtv = training + val, X1 = train, X2 = val, Xt = test
    X, Y, C = reformat(data[:9*N/10,:])
    Xtv, Ytv, Ctv = reformat(data[4*N/5:,:])
    X1, Y1, C1 = reformat(data1)
    X2, Y2, C2 = reformat(data2)
    Xt, Yt, Ct = reformat(datat)

    ncomp = 150
#    pca = RandomizedPCA(n_components=ncomp, whiten=True)
    pca = PCA(n_components=ncomp, whiten=True)
    pca.fit(X) # fit pca to training data
    1/0
    Xnew = pca.transform(Xtv)
    X1new = pca.transform(X1)
    X2new = pca.transform(X2)
    Xtnew = pca.transform(Xt)
    
    eigenfaces = pca.components_.reshape((ncomp, 48, 48))
    titles = ["eigenface %i" % i for i in range(len(eigenfaces))]
    plot_gallery(eigenfaces, titles, 48, 48)
    disp = Xtnew[:12]
    titles = Ct[:12]
    projs = disp.dot(pca.components_)
    plot_gallery(projs.reshape((len(projs), 48, 48)), titles, 48, 48)
    plot_gallery(Xt.reshape((len(Xt), 48, 48)), Ct, 48, 48)
    
    idx = np.arange(len(Xnew))
    valsplit = [(idx[:len(X1)], idx[len(X1):])]
    cvals = [1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
    gamvals = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    param_grid = {'C': cvals,'gamma': gamvals, }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=valsplit)
    clf = clf.fit(Xnew, Ctv)
    print clf.score(X2new, C2)
    print clf.score(Xtnew, Ct)

    grid = make_grid(clf.grid_scores_, len(cvals), len(gamvals))
    print grid
    make_heatmap(grid, cvals, gamvals, "C", "gamma")
