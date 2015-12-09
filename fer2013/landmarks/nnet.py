#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *
from sklearn.decomposition import PCA

import tensorflow.python.platform
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

DIR = os.path.dirname(os.getcwd())
TRFILE = os.path.join(DIR, "trainnew.csv")
TESTFILE = os.path.join(DIR, "test1new.csv")


class Data(object):
    batch_size = 128
    def __init__(self, X, Y):
        self.N = self.batch_size * int(len(X)/self.batch_size)
        self.X = X[:self.N]
        self.Y = Y[:self.N]
        self.cur = 0

    def next_batch(self):
        x = self.X[self.cur:self.cur + self.batch_size]
        y = self.Y[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        self.cur %= self.N
        return x, y

def get_data():
    trdat = np.loadtxt(TRFILE, dtype=np.int, delimiter=',')
    testdat = np.loadtxt(TESTFILE, dtype=np.int, delimiter=',')
    print("loaded data")
    X, Y, _ = reformat(trdat)
    Xt, Yt, _ = reformat(testdat)

    return Xnew, Y, Xtnew, Yt


def main(_):
  # Import data
  M = 100

  # get the extracted features from facial images
  X, Y, Xt, Yt = get_data()
  d = X.shape[1]
  k = Y.shape[1]

  trdat = Data(X, Y)
  testdat = Data(Xt, Yt)

  print("starting tf session")
  sess = tf.InteractiveSession()

  # Create the model
  x = tf.placeholder('float', [None, d], name='x-input')
  W1 = tf.Variable(tf.random_uniform([d, M], -0.5, 0.5), name='w1')
  wd1 = tf.mul(tf.nn.l2_loss(W1), 0.004)
  b1 = tf.Variable(tf.random_uniform([M], -0.5, 0.5), name='b1')
  l1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
  W2 = tf.Variable(tf.random_uniform([M, k], -0.5, 0.5), name='w2')
  wd2 = tf.mul(tf.nn.l2_loss(W2), 0.004)
  b2 = tf.Variable(tf.random_uniform([k], -0.5, 0.5), name='b2')

  # use a name scope to organize nodes in the graph visualizer
  with tf.name_scope('Wx_b') as scope:
    y = tf.nn.softmax(tf.matmul(l1, W2) + b2)

  # Add summary ops to collect data
  w1_hist = tf.histogram_summary('weights1', W1)
  b1_hist = tf.histogram_summary('bias1', b1)
  l1_hist = tf.histogram_summary('layer1', l1)
  w2_hist = tf.histogram_summary('weights2', W2)
  b2_hist = tf.histogram_summary('bias2', b2)
  y_hist = tf.histogram_summary('y', y)

  # Define loss and optimizer
  y_ = tf.placeholder('float', [None, k], name='y-input')
  # More name scopes will clean up the graph representation
  with tf.name_scope('xent') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    ce_summ = tf.scalar_summary('cross entropy', cross_entropy)

  total_loss = tf.add_n([wd1, wd2, cross_entropy], name='total_loss')
  with tf.name_scope('train') as scope:
    train_step = tf.train.GradientDescentOptimizer(
        FLAGS.learning_rate).minimize(total_loss)

  with tf.name_scope('test') as scope:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    accuracy_summary = tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter('/tmp/pca_logs', sess.graph_def)
  tf.initialize_all_variables().run()

  # Train the model, and feed in test data and record summaries every 10 steps

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summary data, and the accuracy
      feed = {x: testdat.X, y_: testdat.Y}
      result = sess.run([merged, accuracy], feed_dict=feed)
      summary_str = result[0]
      acc = result[1]
      writer.add_summary(summary_str, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      batch_xs, batch_ys = trdat.next_batch()
      print(batch_xs.shape, batch_ys.shape)
      feed = {x: batch_xs, y_: batch_ys}
      sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
  tf.app.run()
