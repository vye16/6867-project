#!/usr/bin/env python

from classify import *
import tensorflow as tf


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

    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    X, Y, C = load_features(os.path.join(DATA_DIR, "trainnew.csv"))
    Xt, Yt, Ct = load_features(os.path.join(DATA_DIR, "test2new.csv"))
    print X.shape, Y.shape, C.shape

    M = 100
    init, train_step, x, y, y_ = make_nn(M, X.shape[1], Y.shape[1])
    sess = tf.Session()
    sess.run(init)
    for i in range(5000):
        sess.run(train_step, feed_dict={x: X[:len(X)/2], y_: Y[:len(Y)/2]})
        sess.run(train_step, feed_dict={x: X[len(X)/2:], y_: Y[len(Y)/2:]})
        
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "score on training", sess.run(accuracy, feed_dict={x: X, y_: Y})
    print "score on test", sess.run(accuracy, feed_dict={x: Xt, y_: Yt})
