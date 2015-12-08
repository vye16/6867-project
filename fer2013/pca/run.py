import numpy as np
import os
from sklearn import cross_validation as crossval
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import tensorflow as tf


def reformat(data):
    x = data[:,1:]
    classes = data[:,0].astype(np.int)
    K = max(classes)+1
    N = x.shape[0]
    y = np.zeros((N,K))
    y[np.arange(N),classes] = 1
    return x, y, classes

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

def make_nn(M, d, k, lam1=1e-3, lam2=1e-3):
    x = tf.placeholder("float", [None, d])   
    W1 = tf.Variable(tf.random_uniform([d, M], -0.5, 0.5))
    l1 = tf.nn.sigmoid(tf.matmul(x, W1))
    W2 = tf.Variable(tf.random_uniform([M, k], -0.5, 0.5))
    y = tf.nn.softmax(tf.matmul(l1, W2))
    y_ = tf.placeholder("float", [None, k])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y)) + lam1 * tf.nn.l2_loss(W1) + lam2 * tf.nn.l2_loss(W2)
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    return init, train_step, x, y, y_


if __name__=="__main__":
    DATA_DIR = os.path.dirname(os.getcwd())
    data = np.loadtxt(os.path.join(DATA_DIR, "train.csv"), dtype=np.int, delimiter=",")
    datat = np.loadtxt(os.path.join(DATA_DIR, "test2.csv"), dtype=np.int, delimiter=",")
    N = len(data)
    print N
    data1 = data[8*N/10:9*N/10,:]
    data2 = data[9*N/10:,:]

    # X = pca extraction (includes training but not validation)
    # Xtv = training + val
    # X1 = train
    # X2 = val
    X, Y, C = reformat(data[:9*N/10,:])
    Xtv, Ytv, Ctv = reformat(data[4*N/5:,:])
    X1, Y1, C1 = reformat(data1)
    X2, Y2, C2 = reformat(data2)
    # Xt = test
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
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=valsplit)
    clf = clf.fit(Xnew, Ctv)
    print clf.score(X2new, C2)
    print clf.score(Xtnew, Ct)
    
    
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