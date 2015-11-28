"""Routine for decoding the file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow.python.platform
import tensorflow as tf

DATA_DIR = os.path.dirname(os.getcwd())
#datat = np.loadtxt(os.path.join(DATA_DIR, "test2.csv"), dtype=np.int, delimiter=",")

def reformat(data):
    x = data[:,1:]
    classes = data[:,0].astype(np.int)
    K = max(classes)+1
    N = x.shape[0]
    y = np.zeros((N,K))
    y[np.arange(N),classes] = 1
    return x, y, classes

def read_examples(filename):
  """Reads and parses examples from data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  class Record(object):
    pass

  data = np.loadtxt(os.path.join(DATA_DIR, filename), dtype=np.int, delimiter=",")
  X, Y, C = reformat(data)
  results = []
  for x, y, c in zip(X, Y, C):
      result = Record()
      result.height = 48
      result.width = 48
      result.label = tf.cast(c, tf.int32)
      # reshape image from [height * width] to [height, width].
      depth_major = tf.reshape(x, [result.height, result.width, 1])
      #print(depth_major)
      #return results
      result.image = depth_major
      #yield result
      results.append(result)
  return results

#train_examples = read_examples('train.csv')
#test_examples = read_examples('test1.csv')
train_examples = None
test_examples = None

train_counter = 0
test_counter = 0

def read_example(train=True):
   global train_counter
   global test_counter
   global train_examples
   global test_examples
   if train_examples is None and train:
       train_examples = read_examples('train.csv')
   if test_examples is None:
       test_examples = read_examples('test1.csv')

   if train:
       train_counter = (train_counter + 1) % len(train_examples)
       return train_examples[train_counter]
   else:
       test_counter = (test_counter + 1) % len(test_examples)
       return test_examples[test_counter]
