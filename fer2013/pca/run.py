#!/usr/bin/env oython

from utils import *
from nnet import *
from sklearn import cross_validation as crossval
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


def reformat(data):
    x = data[:,1:]
    classes = data[:,0].astype(np.int)
    K = max(classes)+1
    N = x.shape[0]
    y = np.zeros((N,K))
    y[np.arange(N),classes] = 1
    return x, y, classes

def make_grid(grid_scores, cvals, gamvals):
    cidx = {}
    gidx = {}
    for i, c in enumerate(cvals):
        cidx[c] = i
    for i, g in enumerate(gamvals):
        gidx[g] = i

    for score in gridscores:
        pass


if __name__=="__main__":
    do_nnet = False

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

    pca = RandomizedPCA(n_components=150, whiten=True)
    pca.fit(X) # fit pca to training data
    Xnew = pca.transform(Xtv)
    X1new = pca.transform(X1)
    X2new = pca.transform(X2)
    Xtnew = pca.transform(Xt)
    
    eigenfaces = pca.components_.reshape((150, 48, 48))
    titles = ["eigenface %i" % i for i in range(len(eigenfaces))]
    plot_gallery(eigenfaces, titles, 48, 48)
    disp = Xtnew[:12]
    titles = C1[:12]
    projs = X1new.dot(pca.components_)
    plot_gallery(projs.reshape((len(projs), 48, 48)), titles, 48, 48)
    plot_gallery(Xt.reshape((len(Xt), 48, 48)), C1, 48, 48)
    
    idx = np.arange(len(Xnew))
    valsplit = [(idx[:len(X1)], idx[len(X1):])]
    cvals = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
    gamvals = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    param_grid = {'C': cvals,'gamma': gamvals, }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=valsplit)
    clf = clf.fit(Xnew, Ctv)
    print clf.score(X2new, C2)
    print clf.score(Xtnew, Ct)
    
    
    if do_nnet:
        # Neural net
        init, train_step, x, y, y_ = make_nn(100, X1new.shape[1], Y1.shape[1])
        sess = tf.Session()
        sess.run(init)
        for i in range(5000):
        #     idx = i % N
        #     sess.run(train_step, feed_dict={x: Xnew[idx:idx+1], y_: Y[idx:idx+1]})
            sess.run(train_step, feed_dict={x: Xnew[:len(Xnew)/2], y_: Ytv[:len(Ytv)/2]})
            sess.run(train_step, feed_dict={x: Xnew[len(Xnew)/2:], y_: Ytv[len(Ytv)/2:]})
            
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print sess.run(accuracy, feed_dict={x: X1new, y_: Y1})
        print sess.run(accuracy, feed_dict={x: Xtnew, y_: Yt})
