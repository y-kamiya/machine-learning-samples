from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def main(_):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.int64, [None])

  W1 = weight_variable([5,5,1,32])
  b1 = bias_variable([32])
  x1 = tf.reshape(x, [-1, 28, 28, 1])
  hc1 = tf.nn.relu(conv2d(x1, W1) + b1)
  hp1 = max_pool_2x2(hc1)

  W2 = weight_variable([5,5,32,64])
  b2 = bias_variable([64])
  hc2 = tf.nn.relu(conv2d(hp1, W2) + b2)
  hp2 = max_pool_2x2(hc2)

  W3 = weight_variable([7 * 7 * 64, 256])
  b3 = bias_variable([256])
  hp2_flat = tf.reshape(hp2, [-1, 7 * 7 * 64])
  hf3 = tf.nn.relu(tf.matmul(hp2_flat, W3) + b3)
  
  keep_prob = tf.placeholder(tf.float32)
  hf3_drop = tf.nn.dropout(hf3, keep_prob)

  W4 = weight_variable([256, 10])
  b4 = bias_variable([10])

  y = tf.nn.softmax(tf.matmul(hf3_drop, W4) + b4)

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y,1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100) 
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  print("test accuracy %g" % accuracy.eval(feed_dict={ x: eval_data, y_: eval_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    tf.app.run()
