#!/usr/bin/env python

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

